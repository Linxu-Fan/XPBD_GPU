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

namespace {

// Only rendering data crosses the thread boundary. In particular, never copy the
// dense mass matrix, constraint caches, or contact solver buffers for a frame.
struct RenderFrame
{
    state cloth;
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
        parameters simParams;
        material clothMaterial;
        state simState;

		// Load the cloth mesh and initialize the simulation state.
        importAnMesh("assets/remeshed/skirt1252.obj", clothMesh);
        if (!initializeState(clothMesh, clothMaterial, simState)) {
            std::cerr << "Failed to initialize cloth state\n";
            return 1;
        }

        // Mark two stationary pinned vertices
        for (const std::size_t vertex : {std::size_t{74}, std::size_t{55}}) {
            simState.pinned_positions[vertex] = clothMesh.vertices[vertex];
        }

        std::filesystem::create_directories("output");
        diffxpbd::Viewer viewer(clothMesh);
        const double timestep = simParams.timestep;
        auto initialFrame = std::make_shared<RenderFrame>();
        initialFrame->cloth.position = simState.position;
        initialFrame->cloth.pinned_positions = simState.pinned_positions;
        std::shared_ptr<const RenderFrame> displayed = std::move(initialFrame);
        viewer.draw(clothMesh, displayed->cloth, 0, 0.0, simParams.totalSteps <= 0);

        SimulationWorker simulation;
        simulation.thread = std::thread([&] {
            try {
                for (int step = 0; step < simParams.totalSteps; ++step) {
                    {
                        std::unique_lock<std::mutex> lock(simulation.mutex);
                        simulation.wake.wait(lock, [&] {
                            return simulation.stop || !simulation.paused || simulation.pending_steps > 0;
                        });
                        if (simulation.stop) return;
                        if (simulation.pending_steps > 0) --simulation.pending_steps;
                    }

                    if (!advanceState(clothMesh, clothMaterial, simParams, simState)) {
                        throw std::runtime_error("Simulation failed at timestep " + std::to_string(step));
                    }
                    exportStateAsObj("output/skirt.obj", clothMesh, simState, step);

                    auto frame = std::make_shared<RenderFrame>();
                    frame->cloth.position = simState.position;
                    frame->cloth.pinned_positions = simState.pinned_positions;
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
                            displayed->step * timestep, displayed->step >= simParams.totalSteps);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
