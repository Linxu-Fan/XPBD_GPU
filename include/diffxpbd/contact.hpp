#pragma once

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <optional>
#include <vector>
#include "diffxpbd/kinematic_collider.hpp"

struct mesh;

namespace diffxpbd {

struct ContactSettings
{
    double separation = 0.001; // m, minimum desired cloth-obstacle distance
    double compliance = 1e-7; // m/N
    double ccd_tolerance = 1e-7; // m
    int max_extra_contact_sweeps = 4; // these are additional contact-only sweeps after the main solver finishes
    int max_extra_contact_step_levels = 4; // at most 2^max_extra_contact_step_levels internal substeps
};

enum class ContactFeatureType {
    Unset = 0,
    Vertex = 1,
    Edge = 2,
    Triangle = 3
};

struct ContactPoint
{
    ContactFeatureType feature_type = ContactFeatureType::Unset;

    // The first vertex_count() indices and weights interpolate the point.
    std::array<std::size_t, 3> indices{};
    std::array<double, 3> weights{};

    constexpr std::size_t vertex_count() const noexcept
    {
        return static_cast<std::size_t>(feature_type);
    }
};

struct Contact
{
    ContactPoint cloth;
    ContactPoint obstacle;
    // Unit normal pointing from the obstacle toward the admissible cloth side.
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    // CCD time as a fraction of the substep in [0, 1], when available.
    std::optional<double> time_of_impact;
};

struct ContactEvaluation
{
    // g = n^T(p_{cloth} - p_{obstacle}) - d.
    double gap;
    // Row i is the derivative with respect to the corresponding feature vertex.
    // Only the first 1, 2, or 3 rows are used for a vertex, edge, or triangle feature respectively. Remaining rows are zero.
    Eigen::Matrix3d cloth_jacobian = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d obstacle_jacobian = Eigen::Matrix3d::Zero();
};

// Features, weights and normal are fixed when evaluating these local derivatives.
// It takes an existing contact and evaluates its gap and position derivatives using the supplied current positions.
ContactEvaluation evaluateContact(const Contact& contact,
    const std::vector<Eigen::Vector3d>& cloth_positions,
    const std::vector<Eigen::Vector3d>& obstacle_positions, double separation);

// Inputs must have matching vertex order and valid triangle connectivity.
// Detects vertex-face in both directions and edge-edge, using swept AABB trees.
std::vector<Contact> detectMeshContacts(const mesh& cloth,
    const std::vector<Eigen::Vector3d>& start_positions,
    const std::vector<Eigen::Vector3d>& end_positions,
    const ColliderInput& collider, const ContactSettings& settings,
    const std::vector<Contact>* known_contacts = nullptr);

} // namespace diffxpbd
