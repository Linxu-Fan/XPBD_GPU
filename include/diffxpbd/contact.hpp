#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <vector>

struct mesh;

namespace diffxpbd {

struct ContactSettings
{
    bool self_collision = true;
    double separation = 0.001; // m, desired distance between cloth layers.
    double compliance = 1e-7; // m/N, zero gives hard contact.
    double ccd_tolerance = 1e-7; // m
    int max_extra_contact_sweeps = 4;
    int max_extra_contact_step_levels = 4; // At most 2^levels internal substeps.
};

enum class ContactFeatureType { Unset = 0, Vertex = 1, Edge = 2, Triangle = 3 };

struct ContactPoint
{
    ContactFeatureType feature_type = ContactFeatureType::Unset;
    // The first vertex_count() indices and weights interpolate a cloth point.
    std::array<std::size_t, 3> indices{};
    std::array<double, 3> weights{};
    constexpr std::size_t vertex_count() const noexcept
    {
        return static_cast<std::size_t>(feature_type);
    }
};

struct Contact
{
    ContactPoint first;
    ContactPoint second;
    // Both features index the same cloth. The normal points from second to first,
    // preserving their pre-impact side rather than assigning a global outside.
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double time_of_impact = 0.0; // Fraction of the current substep in [0, 1].
};

struct ContactEvaluation
{
    double gap; // n dot (p_first - p_second) - separation.
    // Row i contains the XYZ gradient for that feature's ith cloth vertex.
    // Only the first vertex_count() rows are used; remaining rows are zero.
    Eigen::Matrix3d first_jacobian = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d second_jacobian = Eigen::Matrix3d::Zero();
};

// Normals and interpolation weights stay fixed during this local evaluation.
ContactEvaluation evaluateContact(const Contact& contact,
    const std::vector<Eigen::Vector3d>& positions, double separation);

// Inputs use valid mesh indices, matching finite position buffers and nonnegative
// contact separation with positive CCD tolerance. BVHs and candidates are rebuilt
// from the supplied endpoints on every call.
// One-shot query. Known feature pairs are skipped; null checks every pair.
std::vector<Contact> detectSelfContacts(const mesh& cloth,
    const std::vector<Eigen::Vector3d>& start_positions,
    const std::vector<Eigen::Vector3d>& end_positions, const ContactSettings& settings,
    const std::vector<Contact>* known_contacts = nullptr);

} // namespace diffxpbd
