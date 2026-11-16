#include "diffxpbd/xpbd.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using Vec3 = Eigen::Vector3d;
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
mesh patch(double height)
{
    mesh m;
    m.vertices = {{-0.2, height, -0.2}, {0.2, height, -0.2}, {0, height, 0.2}};
    m.triangles.push_back({{0, 1, 2}, Vec3::UnitX(), Vec3::UnitZ()});
    return m;
}

// Two disconnected patches exercise two-sided response and substep retries.
void checkSelfContact(bool jacobi, bool pinned)
{
    mesh cloth;
    cloth.vertices = {{-0.2, -0.01, -0.2}, {0.2, -0.01, -0.2}, {0, -0.01, 0.2},
                      {-0.18, 0.01, -0.18}, {0.22, 0.01, -0.18}, {0.02, 0.01, 0.22}};
    cloth.triangles = {{{0, 1, 2}, Vec3::UnitX(), Vec3::UnitZ()},
                       {{3, 4, 5}, Vec3::UnitX(), Vec3::UnitZ()}};
    material mat;
    parameters params;
    params.gravity.setZero();
    params.timestep = 0.01;
    params.totalIterations = 20;
    params.JacobiSolver = jacobi;
    params.contact.compliance = 0.0;
    state s;
    require(initializeState(cloth, mat, s), "Self-contact initialization failed");
    double initial_momentum = 0.0;
    for (std::size_t i = 0; i < s.position.size(); ++i) {
        s.velocity[i] = Vec3(0, i < 3 ? 2.0 : -2.0, 0);
        if (pinned && i < 3) {
            s.pinned_positions[i] = s.position[i];
            s.velocity[i].setZero();
        }
        initial_momentum += s.M(3 * i, 3 * i) * s.velocity[i].y();
    }
    require(advanceState(cloth, mat, params, s), "Self-contact impact failed");
    require(!s.contacts.empty(), "No self-contacts generated");
    double momentum = 0.0;
    for (std::size_t i = 0; i < s.position.size(); ++i) {
        require(s.position[i].allFinite(), "Nonfinite self-contact result");
        momentum += s.M(3 * i, 3 * i) * s.velocity[i].y();
        if (pinned && i < 3)
            require((s.position[i] - cloth.vertices[i]).norm() == 0.0 &&
                    s.velocity[i].isZero(0.0), "Self-contact moved a pin");
    }
    // The patches can rotate, so overlapping world-Y ranges do not imply a
    // crossing. All upper vertices must stay on the original side of the lower
    // triangle's plane. This also rules out a final triangle intersection.
    Vec3 normal = (s.position[1] - s.position[0]).cross(s.position[2] - s.position[0]).normalized();
    if (normal.y() < 0.0) normal = -normal;
    for (std::size_t i = 3; i < 6; ++i)
        require(normal.dot(s.position[i] - s.position[0]) > 0.0,
                "Self-contact patches exchanged sides");
    if (!pinned)
        require(std::abs(momentum - initial_momentum) < 1e-9, "Self-contact lost linear momentum");

    // With the feature choices fixed, both Jacobian blocks must describe the
    // same cloth buffer. Perturbing either feature checks the future adjoint API.
    const auto& contact = s.contacts.front();
    const auto evaluation = diffxpbd::evaluateContact(contact, s.position,
                                                     params.contact.separation);
    constexpr double epsilon = 1e-6;
    for (int side = 0; side < 2; ++side) {
        const auto& feature = side == 0 ? contact.first : contact.second;
        const auto& jacobian = side == 0 ? evaluation.first_jacobian : evaluation.second_jacobian;
        for (std::size_t j = 0; j < feature.vertex_count(); ++j)
            for (int axis = 0; axis < 3; ++axis) {
                auto plus = s.position, minus = s.position;
                plus[feature.indices[j]][axis] += epsilon;
                minus[feature.indices[j]][axis] -= epsilon;
                const double derivative =
                    (diffxpbd::evaluateContact(contact, plus, params.contact.separation).gap -
                     diffxpbd::evaluateContact(contact, minus, params.contact.separation).gap) /
                    (2 * epsilon);
                require(std::abs(derivative - jacobian(j, axis)) < 1e-9,
                        "Self-contact derivative mismatch");
            }
    }
}

void checkSelfFiltering()
{
    auto cloth = patch(0.0);
    diffxpbd::ContactSettings settings;
    require(diffxpbd::detectSelfContacts(cloth, cloth.vertices, cloth.vertices, settings).empty(),
            "Incident features generated self-contact");

    // Two nonincident edges cross while every vertex stays away from the other
    // face interior. Verify edge-edge queries and their canonical ordering.
    mesh edges;
    edges.vertices = {{-1, 0, 0}, {1, 0, 0}, {0, 0, -1},
                      {0, 0.1, -0.5}, {0, 0.1, 0.5}, {1, 0.1, 0}};
    edges.triangles = {{{0, 1, 2}, Vec3::UnitX(), Vec3::UnitZ()},
                      {{3, 4, 5}, Vec3::UnitX(), Vec3::UnitZ()}};
    auto end = edges.vertices;
    for (std::size_t i = 3; i < 6; ++i) end[i].y() = -0.1;
    const auto contacts = diffxpbd::detectSelfContacts(edges, edges.vertices, end, settings);
    bool found_edge = false;
    for (const auto& contact : contacts) {
        if (contact.first.feature_type == diffxpbd::ContactFeatureType::Edge) {
            found_edge = true;
            require(contact.first.indices < contact.second.indices, "Duplicate edge orientation");
        }
    }
    require(found_edge, "Self edge-edge CCD missed crossing");
    require(diffxpbd::detectSelfContacts(edges, edges.vertices, end, settings, &contacts).empty(),
            "Known self-contacts were duplicated");
}



mesh pairOfPatches()
{
    mesh cloth;
    cloth.vertices = {{-0.2, -0.01, -0.2}, {0.2, -0.01, -0.2}, {0, -0.01, 0.2},
                      {-0.18, 0.01, -0.18}, {0.22, 0.01, -0.18}, {0.02, 0.01, 0.22}};
    cloth.triangles = {{{0, 1, 2}, Vec3::UnitX(), Vec3::UnitZ()},
                      {{3, 4, 5}, Vec3::UnitX(), Vec3::UnitZ()}};
    return cloth;
}

void checkFreshDetection()
{
    auto cloth = pairOfPatches();
    const auto base = cloth;
    // More than eight primitives exercises both leaves and internal BVH nodes.
    for (std::size_t copy = 1; copy < 5; ++copy) {
        const auto offset = cloth.vertices.size();
        for (const auto& x : base.vertices) cloth.vertices.push_back(x + Vec3(4.0 * copy, 0, 0));
        for (auto triangle : base.triangles) {
            for (auto& vertex : triangle.indices) vertex += offset;
            cloth.triangles.push_back(triangle);
        }
    }
    diffxpbd::ContactSettings settings;
    for (int pass = 0; pass < 4; ++pass) {
        auto end = cloth.vertices;
        // Alternate distant and crossing endpoints and verify that independently
        // rebuilt BVHs return the same ordered contacts for identical inputs.
        if (pass % 2 == 0)
            for (std::size_t i = 3; i < 6; ++i) end[i].y() = -0.03;
        const auto expected = diffxpbd::detectSelfContacts(cloth, cloth.vertices, end, settings);
        for (int repeat = 0; repeat < 2; ++repeat) {
            const auto actual = diffxpbd::detectSelfContacts(cloth, cloth.vertices, end, settings);
            require(actual.size() == expected.size(), "Fresh detection changed contact count");
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const auto& a = actual[i];
                const auto& b = expected[i];
                require(a.first.feature_type == b.first.feature_type &&
                        a.second.feature_type == b.second.feature_type &&
                        a.first.indices == b.first.indices && a.second.indices == b.second.indices &&
                        a.first.weights == b.first.weights && a.second.weights == b.second.weights &&
                        (a.normal.array() == b.normal.array()).all() &&
                        a.time_of_impact == b.time_of_impact, "Fresh detection changed contact geometry");
            }
        }
    }
}

void checkRollbackAndDisable()
{
    auto cloth = pairOfPatches();
    for (std::size_t i = 0; i < 3; ++i) cloth.vertices[i].y() = 0.0;
    for (std::size_t i = 3; i < 6; ++i) cloth.vertices[i].y() = 0.0005;
    state s;
    material mat;
    parameters params;
    params.gravity.setZero();
    params.contact.max_extra_contact_step_levels = 1;
    require(initializeState(cloth, mat, s), "Rollback initialization failed");
    for (std::size_t i = 0; i < s.position.size(); ++i) s.pinned_positions[i] = s.position[i];
    require(!advanceState(cloth, mat, params, s), "Infeasible pinned layers were accepted");
    for (std::size_t i = 0; i < s.position.size(); ++i)
        require((s.position[i].array() == cloth.vertices[i].array()).all() && s.velocity[i].isZero(0),
                "Failed self-contact trial changed caller state");
    require(s.contacts.empty(), "Failed trial committed contact records");
    params.contact.self_collision = false;
    require(advanceState(cloth, mat, params, s), "Disabled self-contact path failed");
    require(s.contacts.empty(), "Disabled self-contact generated contacts");
}

} // namespace

int main()
{
    try {
        checkFreshDetection();
        checkRollbackAndDisable();
        checkSelfFiltering();
        checkSelfContact(false, false);
        checkSelfContact(true, false);
        checkSelfContact(false, true);
        std::cout << "Self-contact checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
