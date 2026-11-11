#include "diffxpbd/xpbd.hpp"
#include "diffxpbd/build_info.hpp"

#include <exception>
#include <filesystem>
#include <iostream>

int main(int argc, char* argv[])
{
	/////////////////////////////////////////////////////////////////////////////////
	// A simple forward simulation of XPBD
	/////////////////////////////////////////////////////////////////////////////////

	// Step 0: define the simulation state and mesh
	mesh clothMesh;
	parameters simParams;
	material clothMaterial;
	state simState;
	// Configure this with obstacle topology and a prescribed world-space motion sampler.
	std::optional<diffxpbd::KinematicCollider> collider;

	// Step 1: read the mesh
	const auto defaultMeshPath = std::filesystem::u8path(DIFFXPBD_DEFAULT_MESH_PATH);
	const std::filesystem::path meshPath = argc > 1
	    ? std::filesystem::path(argv[1])
	    : defaultMeshPath;
	try {
	    importAnMesh(meshPath, clothMesh);
	} catch (const std::exception& error) {
	    std::cerr << "Failed to load mesh: " << error.what() << '\n';
	    return 1;
	}

	// Step 2: initialize the simulation state
	if (!initializeState(clothMesh, clothMaterial, simState)) {
	    std::cerr << "Failed to initialize cloth state\n";
	    return 1;
	}

	std::error_code meshPathError;
	if (std::filesystem::equivalent(meshPath, defaultMeshPath, meshPathError)) {
	    // Opposite sides of the waistband of skirt1252.obj (zero-based).
	    for (const std::size_t vertex : {std::size_t{74}, std::size_t{55}}) {
	        if (vertex >= clothMesh.vertices.size()) {
	            std::cerr << "The default skirt is missing a waistband pin vertex\n";
	            return 1;
	        }
	        simState.pinned_positions[vertex] = clothMesh.vertices[vertex];
	    }
	}

	// Step 3: run the forward simulation for a number of timesteps
	const int numSteps = simParams.totalSteps;
	const double timestep = simParams.timestep;
	const int numIterations = simParams.totalIterations;
	for (int i = 0; i < numSteps; i++)
	{
		std::cout << "Step " << i << " / " << numSteps << std::endl;
	    if (!calExternalForce(clothMesh, simParams, simState)) {
	        std::cerr << "Failed to calculate external force at timestep " << i << '\n';
	        return 1;
	    }

		std::optional<diffxpbd::ColliderInput> colliderInput;
		if (collider) {
		    const double startTime = i * timestep;
		    const double endTime = (i + 1) * timestep;
		    try {
		        colliderInput.emplace(diffxpbd::ColliderInput{
		            collider->triangles(), collider->sampleStep(startTime, endTime)});
		    } catch (const std::exception& error) {
		        std::cerr << "Failed to sample obstacle at timestep " << i
		                  << ": " << error.what() << '\n';
		        return 1;
		    }
		}
		bool stepSuccess = advanceState(clothMesh, clothMaterial, simParams, simState,
		                                colliderInput ? &*colliderInput : nullptr);
		if (stepSuccess)
		{
			exportStateAsObj("output/skirt.obj", clothMesh, simState, i);
		}
		else
		{
			std::cout << "Simulation failed at timestep " << i << std::endl;
			return 0;
		}
	}



	return 0;

}
