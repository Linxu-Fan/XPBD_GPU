#pragma once

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

namespace diffxpbd {

// stores the obstacle's vertex positions at both ends of one simulation substep
struct ColliderStep
{
	double start_time; // the simulation time at the start of the substep
	double end_time; // the simulation time at the end of the substep
    std::vector<Eigen::Vector3d> start_positions;
    std::vector<Eigen::Vector3d> end_positions;
};

// describes a surface mesh whose motion is prescribed. It stores fixed triangle connectivity 
// and asks a callback where each vertex should be at a given time.
class KinematicCollider
{
public:
    using Triangle = std::array<std::size_t, 3>;
    // Returns world-space positions in the same vertex order at every time.
    using PositionSampler = std::function<std::vector<Eigen::Vector3d>(double)>;

    KinematicCollider(std::size_t vertex_count, std::vector<Triangle> triangles,
                      PositionSampler motion);

    const std::vector<Triangle>& triangles() const noexcept; // returns a read-only reference to the stored triangles without copying them.
	std::vector<Eigen::Vector3d> sample(double time) const; // returns the world-space positions of every vertex in the original vertex order at the given time.
	ColliderStep sampleStep(double start_time, double end_time) const; // returns the world-space positions of every vertex in the original vertex order at both ends of a simulation substep.

private:
    std::size_t vertex_count_; // number of vertices in the obstacle mesh
	std::vector<Triangle> triangles_; // fixed triangle connectivity of the obstacle mesh
    PositionSampler motion_; // Given a time in seconds, it returns the world-space position of every vertex in the original vertex order
};

} // namespace diffxpbd
