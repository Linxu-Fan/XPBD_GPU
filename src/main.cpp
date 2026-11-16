#include "diffxpbd/xpbd.hpp"
#include "diffxpbd/viewer.hpp"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <exception>
#include <filesystem>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

namespace {

// Only rendering data crosses the thread boundary. In particular, never copy the
// dense mass matrix, constraint caches, or contact solver buffers for a frame.
struct RenderFrame
{
    explicit RenderFrame(const std::vector<diffxpbd::KinematicCollider::Triangle>& triangles)
        : collider{triangles, {}} {}

    state cloth;
    diffxpbd::ColliderInput collider;
    int step = 0;
};

// The worker owns the live simulation state. The window thread exchanges commands
// and immutable snapshots under this mutex, then renders AFTER releasing the lock.
// Keeping just the latest snapshot bounds memory and never makes physics wait for
// a slow renderer. Every completed step is still exported to OBJ.
struct SimulationWorker
{
    std::mutex mutex;
    std::condition_variable wake;
    bool stop = false;
    bool paused = false;
    unsigned pending_steps = 0;
    std::shared_ptr<const RenderFrame> latest;
    std::exception_ptr error;
    std::thread thread;

    ~SimulationWorker()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        wake.notify_one();
        // Also runs on exceptions. Join before destroying any captured mesh/state.
        if (thread.joinable()) thread.join();
    }
};

} // namespace

int main()
{
    try {
        // Initialize the state
        mesh clothMesh;
        mesh obstacleMesh;
        parameters simParams;
        material clothMaterial;
        state simState;

		// Load the obstacle mesh and create a kinematic collider with a constant upward velocity.   
        importAnMesh("assets/remeshed/cube.obj", obstacleMesh);
        std::vector<diffxpbd::KinematicCollider::Triangle> triangles;
        for (const auto& triangle : obstacleMesh.triangles) triangles.push_back(triangle.indices);
        const auto obstacleVertexCount = obstacleMesh.vertices.size();
        const Eigen::Vector3d obstacleVelocity(0.0, 0.5, 0.0); // world-space velocity in m/s
        diffxpbd::KinematicCollider collider(obstacleVertexCount, std::move(triangles),
            [restPositions = obstacleMesh.vertices, obstacleVelocity](double time) {
                auto positions = restPositions;
                for (auto& position : positions) {
                    position += obstacleVelocity * time;
                }
                return positions;
            });

		// Load the cloth mesh and initialize the simulation state.
        importAnMesh("assets/remeshed/skirt1252.obj", clothMesh);
        if (!initializeState(clothMesh, clothMaterial, simState)) {
            std::cerr << "Failed to initialize cloth state\n";
            return 1;
        }

        // Mark two stationary pinned vertices
        for (const std::size_t vertex : {std::size_t{74}, std::size_t{55}}) {
            if (vertex >= clothMesh.vertices.size()) {
                std::cerr << "The skirt is missing a waistband pin vertex\n";
                return 1;
            }
            simState.pinned_positions[vertex] = clothMesh.vertices[vertex];
        }

        std::filesystem::create_directories("output");
        diffxpbd::Viewer viewer(clothMesh);
        const double timestep = simParams.timestep;
        auto initialFrame = std::make_shared<RenderFrame>(collider.triangles());
        initialFrame->cloth.position = simState.position;
        initialFrame->cloth.pinned_positions = simState.pinned_positions;
        // Initial frame is at t=0, rather than showing the obstacle one step ahead.
        const auto initial = collider.sample(0.0);
        initialFrame->collider.step = {0.0, 0.0, initial, initial};
        std::shared_ptr<const RenderFrame> displayed = std::move(initialFrame);
        viewer.draw(clothMesh, displayed->cloth, 0, 0.0, simParams.totalSteps <= 0,
                    &displayed->collider);

        SimulationWorker simulation;
        simulation.thread = std::thread([&] {
            try {
                diffxpbd::ColliderInput input{collider.triangles(), {}};
                for (int step = 0; step < simParams.totalSteps; ++step) {
                    {
                        std::unique_lock<std::mutex> lock(simulation.mutex);
                        simulation.wake.wait(lock, [&] {
                            return simulation.stop || !simulation.paused || simulation.pending_steps > 0;
                        });
                        if (simulation.stop) return;
                        if (simulation.pending_steps > 0) --simulation.pending_steps;
                    }

                    const double startTime = step * timestep;
                    const double endTime = (step + 1) * timestep;
                    input.step = collider.sampleStep(startTime, endTime);
                    if (!advanceState(clothMesh, clothMaterial, simParams, simState, &input)) {
                        throw std::runtime_error("Simulation failed at timestep " + std::to_string(step));
                    }
                    exportStateAsObj("output/skirt.obj", clothMesh, simState, step);
                    exportStateAsObj("output/cube.obj", obstacleMesh, input.step.end_positions, step);

                    auto frame = std::make_shared<RenderFrame>(collider.triangles());
                    frame->cloth.position = simState.position;
                    frame->cloth.pinned_positions = simState.pinned_positions;
                    frame->collider.step = input.step;
                    frame->step = step + 1;
                    {
                        std::lock_guard<std::mutex> lock(simulation.mutex);
                        simulation.latest = std::move(frame);
                    }
                }
            } catch (...) {
                // Propagate failures to main instead of terminating the process
                // with an uncaught exception on the background thread.
                std::lock_guard<std::mutex> lock(simulation.mutex);
                simulation.error = std::current_exception();
            }
        });

        while (viewer.pollEvents()) {
            const bool paused = viewer.isPaused();
            const bool singleStep = viewer.takeSingleStep();
            std::shared_ptr<const RenderFrame> ready;
            std::exception_ptr error;
            bool commandChanged = false;
            {
                std::lock_guard<std::mutex> lock(simulation.mutex);
                commandChanged = simulation.paused != paused || singleStep;
                simulation.paused = paused;
                if (!paused) simulation.pending_steps = 0;
                if (singleStep) ++simulation.pending_steps;
                ready = std::move(simulation.latest);
                error = simulation.error;
            }
            if (commandChanged) simulation.wake.notify_one();
            if (error) std::rethrow_exception(error);
            if (ready) {
                // Release the previous snapshot outside the publication lock.
                displayed = ready;
            }

            // No continuous redraw while physics is busy, paused, or finished.
            // Camera, resize, and expose events can redraw the last completed
            // snapshot without waiting for another simulation step.
            if (ready || viewer.needsRedraw()) {
                viewer.draw(clothMesh, displayed->cloth, displayed->step,
                            displayed->step * timestep, displayed->step >= simParams.totalSteps,
                            &displayed->collider);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
