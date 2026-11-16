#pragma once

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <optional>
#include <memory>
#include <vector>
#include "diffxpbd/kinematic_collider.hpp"

struct mesh;

namespace diffxpbd {

struct ContactSettings
{
    // Enable conservative candidate reuse and obstacle proximity scheduling.
    // Self-contact retains the original full CCD narrow phase for layer ordering.
    // Disable to compare with the full per-sweep reference search.
    bool cache_collision_queries = true;
    // Endpoint displacement allowed inside a candidate envelope, in meters.
    // Exceeding this budget rebuilds candidates before the next projection.
    double collision_cache_padding = 0.002;
    bool self_collision = true; // Enable cloth vertex-face and edge-edge self-contact.

    double separation = 0.001; // m, desired distance for both obstacle and self-contact
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
    // For self-contact, both features index the cloth position buffer. The second
    // feature keeps its legacy name to preserve existing obstacle-contact callers.
    bool self_contact = false;
    // Unit normal from the second feature toward the admissible first-feature side.
    // For self-contact there is no global outside; CCD preserves the pre-impact side.
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    // Absent for proximity-only potential contacts that did not run CCD.
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

// Scratch acceleration data for a fixed cloth topology. Keep one instance across
// solver sweeps/retries. Candidate envelopes contain both trajectory endpoints,
// and are rebuilt when motion leaves their displacement budget. A separating
// plane may skip CCD only after proving separation over the ENTIRE current
// linear trajectory. Discovery uses full CCD initially and on envelope escape,
// then obstacle proximity refreshes inside the envelope. Endpoint contacts use
// endpoint geometry and have no impact time, so this changes the discrete forward
// contact response. Self-contact keeps full CCD. A null known_contacts pointer
// always requests full swept validation, irrespective of cached schedule.
// This cache contains no physical state and is not part of the adjoint state.
// Use one instance per simulation thread; the bound mesh topology must stay fixed.
class ContactDetector
{
public:
    explicit ContactDetector(const mesh& cloth);
    ~ContactDetector();
    // Reset CCD scheduling for a new physics substep, including retries. BVHs
    // remain allocated, but the first discovery query does full swept detection.
    void beginSubstep();
    ContactDetector(const ContactDetector&) = delete;
    ContactDetector& operator=(const ContactDetector&) = delete;

    std::vector<Contact> detectMeshContacts(
        const std::vector<Eigen::Vector3d>& start_positions,
        const std::vector<Eigen::Vector3d>& end_positions,
        const ColliderInput& collider, const ContactSettings& settings,
        const std::vector<Contact>* known_contacts = nullptr);

    std::vector<Contact> detectSelfContacts(
        const std::vector<Eigen::Vector3d>& start_positions,
        const std::vector<Eigen::Vector3d>& end_positions,
        const ContactSettings& settings,
        const std::vector<Contact>* known_contacts = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Inputs must have matching vertex order and valid triangle connectivity.
// Detects vertex-face in both directions and edge-edge, using swept AABB trees.
std::vector<Contact> detectMeshContacts(const mesh& cloth,
    const std::vector<Eigen::Vector3d>& start_positions,
    const std::vector<Eigen::Vector3d>& end_positions,
    const ColliderInput& collider, const ContactSettings& settings,
    const std::vector<Contact>* known_contacts = nullptr);

// Swept self-contact queries exclude incident features and duplicate edge pairs.
// Both endpoint buffers describe the same cloth, with stable vertex ordering.
std::vector<Contact> detectSelfContacts(const mesh& cloth,
    const std::vector<Eigen::Vector3d>& start_positions,
    const std::vector<Eigen::Vector3d>& end_positions, const ContactSettings& settings,
    const std::vector<Contact>* known_contacts = nullptr);

} // namespace diffxpbd
