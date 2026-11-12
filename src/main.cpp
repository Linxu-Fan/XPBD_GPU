#include "diffxpbd/xpbd.hpp"
#include "diffxpbd/viewer.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

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
        int step = 0;
        diffxpbd::ColliderInput colliderInput{collider.triangles(), collider.sampleStep(0.0, timestep)};
        viewer.draw(clothMesh, simState, step, 0.0, false,
                    &colliderInput);
        while (viewer.pollEvents()) {
            const bool finished = step >= simParams.totalSteps;
            if (finished || !viewer.shouldAdvance()) {
                viewer.draw(clothMesh, simState, step, step * timestep, finished,
                            &colliderInput);
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
                continue;
            }

            const double startTime = step * timestep;
            const double endTime = (step + 1) * timestep;
            colliderInput.step = collider.sampleStep(startTime, endTime);
            if (!advanceState(clothMesh, clothMaterial, simParams, simState,
                              &colliderInput)) {
                std::cerr << "Simulation failed at timestep " << step << '\n';
                return 1;
            }
            exportStateAsObj("output/skirt.obj", clothMesh, simState, step);
            exportStateAsObj("output/cube.obj", obstacleMesh, colliderInput.step.end_positions, step);
            ++step;
            viewer.draw(clothMesh, simState, step, step * timestep,
                        step >= simParams.totalSteps, &colliderInput);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
