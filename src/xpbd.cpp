#include "diffxpbd/xpbd.hpp"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <tuple>

// Forward position prediction, constraint projection, and velocity updates go here.

void importAnMesh(const std::filesystem::path& mesh_path, mesh& m)
{
    const auto utf8_path = mesh_path.u8string();
    const std::string path_text(utf8_path.begin(), utf8_path.end());
    std::ifstream input(mesh_path);
    if (!input) {
        throw std::runtime_error("Cannot open OBJ file: " + path_text);
    }

    mesh imported;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line_number == 1 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) {
            line.erase(0, 3);
        }
        if (const auto comment = line.find('#'); comment != std::string::npos) {
            line.erase(comment);
        }

        std::istringstream record(line);
        record.imbue(std::locale::classic());
        std::string kind;
        if (!(record >> kind)) {
            continue;
        }
        if (kind == "f") {
            mesh_triangle triangle{{}, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
            const auto invalid_face = [&]() {
                return std::runtime_error("Invalid OBJ triangle in " + path_text +
                                          " at line " + std::to_string(line_number));
            };
            for (auto& index : triangle.indices) {
                std::string token;
                if (!(record >> token)) {
                    throw invalid_face();
                }
                std::istringstream field(token.substr(0, token.find('/')));
                field.imbue(std::locale::classic());
                std::ptrdiff_t obj_index;
                if (!(field >> obj_index) || !field.eof() || obj_index == 0) {
                    throw invalid_face();
                }
                const auto count = static_cast<std::ptrdiff_t>(imported.vertices.size());
                const auto resolved = obj_index < 0 ? count + obj_index : obj_index - 1;
                if (resolved < 0) {
                    throw invalid_face();
                }
                index = static_cast<std::size_t>(resolved);
            }
            std::string extra;
            if (record >> extra) {
                throw invalid_face();
            }
            imported.triangles.push_back(std::move(triangle));
            continue;
        }
        if (kind != "v") {
            continue;
        }

        Eigen::Vector3d vertex;
        for (Eigen::Index axis = 0; axis < vertex.size(); ++axis) {
            std::string token;
            record >> token;
            std::istringstream coordinate(token);
            coordinate.imbue(std::locale::classic());
            if (!(coordinate >> vertex[axis]) || !coordinate.eof() ||
                !std::isfinite(vertex[axis])) {
                throw std::runtime_error("Invalid OBJ vertex in " + path_text +
                                         " at line " + std::to_string(line_number));
            }
        }
        imported.vertices.push_back(vertex);
    }

    if (!input.eof() || input.bad()) {
        throw std::runtime_error("Error reading OBJ file: " + path_text);
    }
    if (imported.vertices.empty()) {
        throw std::runtime_error("OBJ file contains no vertices: " + path_text);
    }

    for (auto& triangle : imported.triangles) {
        const auto [a, b, c] = triangle.indices;
        if (a >= imported.vertices.size() || b >= imported.vertices.size() ||
            c >= imported.vertices.size()) {
            throw std::runtime_error("OBJ triangle vertex index out of range in " + path_text);
        }

        const Eigen::Vector3d ab = imported.vertices[b] - imported.vertices[a];
        const Eigen::Vector3d ac = imported.vertices[c] - imported.vertices[a];
        Eigen::Vector3d normal = ab.stableNormalized().cross(ac.stableNormalized());
        const double normal_length = normal.norm();
        if (!std::isfinite(normal_length) || normal_length <= 1e-12) {
            throw std::runtime_error("Degenerate OBJ triangle in " + path_text);
        }
        normal /= normal_length;

        const Eigen::Vector3d grain = -Eigen::Vector3d::UnitY();
        triangle.warp = grain - normal.dot(grain) * normal;
        if (triangle.warp.squaredNorm() <= 1e-12) {
            const Eigen::Vector3d fallback = Eigen::Vector3d::UnitX();
            triangle.warp = fallback - normal.dot(fallback) * normal;
        }
        triangle.warp.normalize();
        triangle.weft = normal.cross(triangle.warp).normalized();
    }

    imported.velocities.resize(imported.vertices.size(), Eigen::Vector3d::Zero());
    m = std::move(imported);
}

namespace {

template <std::size_t N>
diffxpbd::Veci<static_cast<int>(3 * N)> positionDofIndices(
    const std::array<std::size_t, N>& vertices)
{
    diffxpbd::Veci<static_cast<int>(3 * N)> indices;
    for (std::size_t local = 0; local < N; ++local) {
        const auto first_dof = static_cast<int>(3 * vertices[local]);
        indices.template segment<3>(static_cast<Eigen::Index>(3 * local))
            << first_dof, first_dof + 1, first_dof + 2;
    }
    return indices;
}

constexpr double two_pi = 6.28318530717958647692;

std::optional<double> signedDihedral(const std::vector<Eigen::Vector3d>& positions,
                                     const bending_hinge& hinge,
                                     diffxpbd::Mat112* derivatives = nullptr)
{
    const auto [a, b] = hinge.edge;
    const auto [c, d] = hinge.opposite;
    const Eigen::Vector3d ab = positions[b] - positions[a];
    const Eigen::Vector3d ac = positions[c] - positions[a];
    const Eigen::Vector3d ad = positions[d] - positions[a];
    const Eigen::Vector3d edge = ab.stableNormalized();
    Eigen::Vector3d first_normal = edge.cross(ac.stableNormalized());
    Eigen::Vector3d second_normal = ad.stableNormalized().cross(edge);
    const double first_length = first_normal.norm();
    const double second_length = second_normal.norm();
    if (!std::isfinite(first_length) || !std::isfinite(second_length) ||
        first_length <= 1e-12 || second_length <= 1e-12) {
        return std::nullopt;
    }

    first_normal /= first_length;
    second_normal /= second_length;
    if (derivatives) {
        const double edge_length = ab.stableNorm();
        const double first_height = ac.stableNorm() * first_length;
        const double second_height = ad.stableNorm() * second_length;
        const Eigen::Vector3d dc = -first_normal / first_height;
        const Eigen::Vector3d dd = -second_normal / second_height;
        // The projections along the edge determine how its endpoints share the gradient.
        const double tc = edge.dot(ac) / edge_length;
        const double td = edge.dot(ad) / edge_length;
        derivatives->segment<3>(0) = ((tc - 1.0) * dc + (td - 1.0) * dd).transpose();
        derivatives->segment<3>(3) = (-tc * dc - td * dd).transpose();
        derivatives->segment<3>(6) = dc.transpose();
        derivatives->segment<3>(9) = dd.transpose();
    }
    return std::atan2(edge.dot(first_normal.cross(second_normal)),
                      first_normal.dot(second_normal));
}

} // namespace

void exportStateAsObj(const std::filesystem::path& mesh_path, const mesh& m, const state& s,
                      std::size_t timestep)
{
    const auto& positions = s.position;
    if (positions.size() != m.vertices.size() || positions.empty() || m.triangles.empty()) {
        throw std::runtime_error("OBJ export requires simulated positions matching a triangle mesh");
    }
    auto filename = mesh_path.stem();
    filename += "_" + std::to_string(timestep);
    filename += mesh_path.has_extension() ? mesh_path.extension() : std::filesystem::path(".obj");
    const auto output_path = mesh_path.parent_path() / filename;
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("Cannot open OBJ for writing: " + output_path.u8string());
    }
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto& position : positions) {
        output << "v " << position.x() << ' ' << position.y() << ' ' << position.z() << '\n';
    }
    for (const auto& triangle : m.triangles) {
        const auto [a, b, c] = triangle.indices;
        output << "f " << a + 1 << ' ' << b + 1 << ' ' << c + 1 << '\n';
    }
    output.close();
    if (!output) {
        throw std::runtime_error("Failed to write OBJ: " + output_path.u8string());
    }
}

bool initializeState(mesh& m, const material& mat, state& s)
{
    diffxpbd::Mat33 membrane_stiffness;
    membrane_stiffness << mat.C00, mat.C01, 0.0,
                          mat.C01, mat.C11, 0.0,
                          0.0,     0.0,     mat.C22;

    const Eigen::LLT<diffxpbd::Mat33> stiffness_factorization(membrane_stiffness);
    if (stiffness_factorization.info() != Eigen::Success) {
        return false;
    }
    const diffxpbd::Mat33 material_compliance =
        stiffness_factorization.solve(diffxpbd::Mat33::Identity());

    const double areal_density = mat.density * mat.thickness;
    constexpr auto max_vertices =
        static_cast<std::size_t>((std::numeric_limits<int>::max() - 2) / 3) + 1;
    if (m.vertices.empty() || m.vertices.size() > max_vertices || m.triangles.empty() ||
        !(mat.density > 0.0) || !(mat.thickness > 0.0) ||
        !std::isfinite(areal_density) || areal_density <= 0.0) {
        return false;
    }

    const auto vertex_count = static_cast<Eigen::Index>(m.vertices.size());
    Eigen::VectorXd vertex_masses = Eigen::VectorXd::Zero(vertex_count);
    std::vector<diffxpbd::Mat22> rest_inverses;
    rest_inverses.reserve(m.triangles.size());
    std::vector<diffxpbd::Mat33> membrane_compliance;
    membrane_compliance.reserve(m.triangles.size());
    std::vector<diffxpbd::Vec9i> membrane_indices;
    membrane_indices.reserve(m.triangles.size());
    struct EdgeFaces
    {
        std::array<std::size_t, 2> opposite{};
        std::size_t count = 0;
    };
    std::map<std::array<std::size_t, 2>, EdgeFaces> edge_faces;

    for (const auto& triangle : m.triangles) {
        const auto [a, b, c] = triangle.indices;
        if (a >= m.vertices.size() || b >= m.vertices.size() ||
            c >= m.vertices.size()) {
            return false;
        }

        const Eigen::Vector3d ab = m.vertices[b] - m.vertices[a];
        const Eigen::Vector3d ac = m.vertices[c] - m.vertices[a];
        const double area = 0.5 * ab.cross(ac).stableNorm();
        const double mass_share = (area / 3.0) * areal_density;
        if (!std::isfinite(mass_share) || mass_share <= 0.0) {
            return false;
        }

        const diffxpbd::Mat33 compliance = material_compliance / area;
        membrane_compliance.push_back(compliance);

        diffxpbd::Mat22 Dm;
        Dm << triangle.warp.dot(ab), triangle.warp.dot(ac),
              triangle.weft.dot(ab), triangle.weft.dot(ac);
        const double determinant = Dm.determinant();
        if (!std::isfinite(determinant) || determinant == 0.0) {
            return false;
        }
        const diffxpbd::Mat22 inverse = Dm.inverse();
        rest_inverses.push_back(inverse);
        membrane_indices.push_back(positionDofIndices(triangle.indices));

        for (std::size_t edge = 0; edge < triangle.indices.size(); ++edge) {
            const auto from = triangle.indices[edge];
            const auto to = triangle.indices[(edge + 1) % triangle.indices.size()];
            auto& faces = edge_faces[{std::min(from, to), std::max(from, to)}];
            if (faces.count < faces.opposite.size()) {
                faces.opposite[faces.count] =
                    triangle.indices[(edge + 2) % triangle.indices.size()];
            }
            ++faces.count;
        }

        for (const auto index : triangle.indices) {
            vertex_masses[static_cast<Eigen::Index>(index)] += mass_share;
        }
    }

    if (!vertex_masses.allFinite() || (vertex_masses.array() <= 0.0).any()) {
        return false;
    }
    Eigen::VectorXd dof_masses(3 * vertex_count);
    for (Eigen::Index vertex = 0; vertex < vertex_count; ++vertex) {
        dof_masses.segment<3>(3 * vertex).setConstant(vertex_masses[vertex]);
    }

    std::vector<bending_hinge> bending_hinges;
    std::vector<diffxpbd::Vec12i> bending_indices;
    for (const auto& [edge, faces] : edge_faces) {
        if (faces.count != 2) {
            continue;
        }
        bending_hinge hinge{edge, faces.opposite, 0.0};
        const auto angle = signedDihedral(m.vertices, hinge);
        if (!angle) {
            return false;
        }
        hinge.rest_angle = *angle;
        bending_hinges.push_back(hinge);
        const std::array<std::size_t, 4> vertices{
            hinge.edge[0], hinge.edge[1], hinge.opposite[0], hinge.opposite[1]};
        bending_indices.push_back(positionDofIndices(vertices));
    }
    const auto hinge_count = bending_hinges.size();
    const double hinge_compliance = hinge_count == 0 ? 0.0 : 1.0 / mat.b;
    if (hinge_count != 0 && (!std::isfinite(hinge_compliance) || hinge_compliance <= 0.0)) {
        return false;
    }
    std::vector<double> bending_compliance(hinge_count, hinge_compliance);
    std::vector<Eigen::Vector3d> membrane_multipliers(
        m.triangles.size(), Eigen::Vector3d::Zero());
    std::vector<double> bending_multipliers(hinge_count, 0.0);
    std::vector<Eigen::Vector3d> membrane_constraints(m.triangles.size(), Eigen::Vector3d::Zero());
    std::vector<double> bending_constraints(hinge_count, 0.0);

    std::vector<Eigen::Vector3d> velocity(
        m.vertices.size(), Eigen::Vector3d::Zero());

    Eigen::MatrixXd mass_matrix = dof_masses.asDiagonal();
    auto positions = m.vertices;
    auto predicted_positions = positions;
    s.M = std::move(mass_matrix);
    s.Dm_inv = std::move(rest_inverses);
    s.position = std::move(positions);
    s.position_predicted = std::move(predicted_positions);
    s.pinned_positions.assign(s.position.size(), std::nullopt);
	s.velocity = std::move(velocity);
    s.multiplier_membrane = std::move(membrane_multipliers);
    s.multiplier_bending = std::move(bending_multipliers);
    s.contacts.clear();
    s.multiplier_contact.clear();
    s.contact_substeps = 1;
    s.constraint_membrane = std::move(membrane_constraints);
    s.constraint_bending = std::move(bending_constraints);
    m.bending_hinges = std::move(bending_hinges);
    s.compliance_membrane = std::move(membrane_compliance);
    s.compliance_bending = std::move(bending_compliance);
    m.indices_membrane = std::move(membrane_indices);
    m.indices_bending = std::move(bending_indices);
    return true;
}

bool calExternalForce(const mesh& m, const parameters& params, state& s)
{
    const auto vertex_count = static_cast<Eigen::Index>(m.vertices.size());
    const Eigen::Index dof_count = 3 * vertex_count;


    const auto masses = s.M.diagonal();


    s.f_ext.resize(m.vertices.size());
    for (Eigen::Index vertex = 0; vertex < vertex_count; ++vertex) {
        s.f_ext[static_cast<std::size_t>(vertex)] =
            masses.segment<3>(3 * vertex).cwiseProduct(params.gravity);
    }
    return true;
}

namespace {

template <std::size_t N>
diffxpbd::Vecd<static_cast<int>(3 * N)> localMassDiagonal(
    const Eigen::MatrixXd& mass_matrix,
    const std::array<std::size_t, N>& vertices)
{
    diffxpbd::Vecd<static_cast<int>(3 * N)> masses;
    for (std::size_t local = 0; local < N; ++local) {
        const auto first_dof = 3 * static_cast<Eigen::Index>(vertices[local]);
        masses.template segment<3>(static_cast<Eigen::Index>(3 * local)) =
            mass_matrix.diagonal().segment<3>(first_dof);
    }
    return masses;
}

diffxpbd::Mat32 membraneDeformationGradient(const std::vector<Eigen::Vector3d>& positions,
                                             const std::array<std::size_t, 3>& vertices,
                                             const diffxpbd::Mat22& rest_inverse)
{
    const auto [a, b, c] = vertices;


    diffxpbd::Mat32 Ds;
    Ds.col(0) = positions[b] - positions[a];
    Ds.col(1) = positions[c] - positions[a];
    return Ds * rest_inverse;
}

Eigen::Vector3d membraneStrain(const diffxpbd::Mat32& F)
{
    const diffxpbd::Mat22 E = 0.5 * (F.transpose() * F - diffxpbd::Mat22::Identity());
    const Eigen::Vector3d constraint(E(0, 0), E(1, 1), 2.0 * E(0, 1));

    return constraint;
}

} // namespace


void calMembraneConstraint_i(const std::vector<Eigen::Vector3d>& positions, const mesh& m, const state& s,
                             int index, Eigen::Vector3d& constraint,
                             diffxpbd::Mat39& constraint_derivatives,
                             diffxpbd::Mat99& constraint_mass_matrix)
{

    const auto triangle_index = static_cast<std::size_t>(index);
    const auto& vertices = m.triangles[triangle_index].indices;

    const auto& rest_inverse = s.Dm_inv[triangle_index];
    const auto F = membraneDeformationGradient(positions, vertices, rest_inverse);
    const auto strain = membraneStrain(F);

    // Each row maps the motion of one vertex to the two columns of F.
    diffxpbd::Mat32 shape_gradients;
    shape_gradients.row(0) = -rest_inverse.colwise().sum();
    shape_gradients.row(1) = rest_inverse.row(0);
    shape_gradients.row(2) = rest_inverse.row(1);

    diffxpbd::Mat39 jacobian;
    for (std::size_t local = 0; local < vertices.size(); ++local) {
        const auto column = static_cast<Eigen::Index>(3 * local);
        const double du = shape_gradients(local, 0);
        const double dv = shape_gradients(local, 1);
        jacobian.block<1, 3>(0, column) = du * F.col(0).transpose();
        jacobian.block<1, 3>(1, column) = dv * F.col(1).transpose();
        jacobian.block<1, 3>(2, column) = (du * F.col(1) + dv * F.col(0)).transpose();

    }

    const auto masses = localMassDiagonal(s.M, vertices);
    constraint = strain;
    constraint_derivatives = jacobian;
    constraint_mass_matrix = masses.asDiagonal();
}


void calBendingConstraint_i(const std::vector<Eigen::Vector3d>& positions, const mesh& m, const state& s,
                            int index, double& constraint,
                            diffxpbd::Mat112& constraint_derivatives,
                            diffxpbd::Mat1212& constraint_mass_matrix)
{

    const auto& hinge = m.bending_hinges[static_cast<std::size_t>(index)];
    diffxpbd::Mat112 jacobian;
    const auto angle = signedDihedral(positions, hinge, &jacobian);
    if (!angle) throw std::runtime_error("Degenerate triangle in bending constraint");

    const std::array<std::size_t, 4> vertices{
        hinge.edge[0], hinge.edge[1], hinge.opposite[0], hinge.opposite[1]};
    const auto masses = localMassDiagonal(s.M, vertices);

    constraint = std::remainder(*angle - hinge.rest_angle, two_pi);
    constraint_derivatives = jacobian;
    constraint_mass_matrix = masses.asDiagonal();
}

namespace {

/*
    The contact is refreshed every sweep. The contact detected earlier is kept in later sweeps.
	a) if the contact is still active, the multiplier is kept and updated.
	b) if the contact is no longer active, the multiplier becomes 0.
	c) if a new contact is detected, the multiplier is initialized to 0.
    d) 
*/
bool advanceStateOnce(const mesh& m, const material& mat, const parameters& params, state& s)
{
    const double dt2 = params.timestep * params.timestep;

    // Masses and topology were validated by initializeState.
    calExternalForce(m, params, s);
    const auto masses = s.M.diagonal();

    const auto maskPinnedInverseMasses = [&s](auto& inv_mass, const auto& indices) {
        for (Eigen::Index dof = 0; dof < indices.size(); dof += 3) {
            if (s.pinned_positions[indices[dof] / 3]) {
                inv_mass.template segment<3>(dof).setZero();
            }
        }
    };

	// update predicted positions based on current positions, velocities, and external forces
    for (int i = 0; i < s.position_predicted.size(); i++)
    {
        if (const auto& anchor = s.pinned_positions[i]) {
            s.position_predicted[i] = *anchor;
            continue;
        }
        const Eigen::Vector3d acceleration = s.f_ext[i].cwiseQuotient(masses.segment<3>(3 * i));
        s.position_predicted[i] = s.position[i] + params.timestep * s.velocity[i] +
                                  dt2 * acceleration;
    }
	s.position_predicted_during_iteration = s.position_predicted;
	s.position_predicted_after_iteration = s.position_predicted;

    // initialize the multiplier
    std::fill(s.multiplier_membrane.begin(), s.multiplier_membrane.end(),
              Eigen::Vector3d::Zero());
    std::fill(s.multiplier_bending.begin(), s.multiplier_bending.end(), 0.0);
    s.contacts.clear();
    s.multiplier_contact.clear();
    s.contact_substeps = 1;


    using ContactKey = std::tuple<int, std::array<std::size_t, 3>,
                                  int, std::array<std::size_t, 3>>;
    std::map<ContactKey, std::size_t> contact_indices;
    // Geometry stays fixed once a self-contact is introduced. Both feature
    // gradients are retained for future local derivatives and adjoint replay.
    const auto detectContacts = [&](const auto& settings, const auto* known) {
        if (!settings.self_collision) return std::vector<diffxpbd::Contact>{};
        return diffxpbd::detectSelfContacts(m, s.position,
            s.position_predicted_during_iteration, settings, known);
    };
    const auto refreshContacts = [&]() {
        auto detected = detectContacts(params.contact, &s.contacts);
        for (auto& contact : detected) {
            const ContactKey key{static_cast<int>(contact.first.feature_type), contact.first.indices,
                                 static_cast<int>(contact.second.feature_type), contact.second.indices};
            if (contact_indices.emplace(key, s.contacts.size()).second) {
                s.contacts.push_back(std::move(contact));
                s.multiplier_contact.push_back(0.0);
            }
        }
    };
    // Keep the original impact side when the same pair is found after projection.
    refreshContacts();
    const auto computeContactConstraint = [&]() {
        refreshContacts();
        const double alpha_tilde = params.contact.compliance / dt2;
        for (std::size_t i = 0; i < s.contacts.size(); ++i) {
            const auto& contact = s.contacts[i];
            const auto evaluation = diffxpbd::evaluateContact(contact,
                s.position_predicted_during_iteration, params.contact.separation);
            double inverse_effective_mass = 0.0;
            std::array<Eigen::Vector3d, 3> directions;
            for (std::size_t j = 0; j < contact.first.vertex_count(); ++j) {
                const auto vertex = contact.first.indices[j];
                directions[j].setZero();
                if (!s.pinned_positions[vertex]) {
                    directions[j] = evaluation.first_jacobian.row(j).transpose().cwiseQuotient(
                        masses.segment<3>(3 * vertex));
                    inverse_effective_mass += evaluation.first_jacobian.row(j).dot(directions[j]);
                }
            }
            // Self-contact transfers equal and opposite impulses between two
            // dynamic features. Include BOTH sides in J M^-1 J^T and in scatter.
            // Incident features were filtered, so no vertex appears on both sides.
            std::array<Eigen::Vector3d, 3> other_directions;
            for (std::size_t j = 0; j < contact.second.vertex_count(); ++j) {
                const auto vertex = contact.second.indices[j];
                other_directions[j].setZero();
                if (!s.pinned_positions[vertex]) {
                    other_directions[j] = evaluation.second_jacobian.row(j).transpose().cwiseQuotient(
                        masses.segment<3>(3 * vertex));
                    inverse_effective_mass += evaluation.second_jacobian.row(j).dot(other_directions[j]);
                }
            }

            // An immovable, violated contact is infeasible. Let the transactional
            // caller retry or report failure rather than silently accepting it.
            if (inverse_effective_mass == 0.0) {
                if (evaluation.gap < -params.contact.ccd_tolerance) return false;
                continue;
            }
            double& multiplier = s.multiplier_contact[i];
            const double increment = (-evaluation.gap - alpha_tilde * multiplier) /
                                     (inverse_effective_mass + alpha_tilde);
            // Unilateral contact can push but cannot pull. A separating pair
            // releases its accumulated multiplier toward zero. Apply only the
            // multiplier change, so repeated sweeps do not reapply old impulses.
            const double next_multiplier = std::max(0.0, multiplier + increment);
            for (std::size_t j = 0; j < contact.first.vertex_count(); ++j) {
                auto& position = s.position_predicted_during_iteration[contact.first.indices[j]];
                position += directions[j] * (next_multiplier - multiplier);
            }
            for (std::size_t j = 0; j < contact.second.vertex_count(); ++j) {
                auto& position = s.position_predicted_during_iteration[contact.second.indices[j]];
                position += other_directions[j] * (next_multiplier - multiplier);
            }

            multiplier = next_multiplier;
        }
        return true;
    };

    // perform constraint projection per iteration
    if (!params.JacobiSolver) // Gauss-Seidel style projection
    {
        int iteration = 0;
        while (iteration < params.totalIterations)
        {
            // compute membrane constraint
            for (int i = 0; i < s.multiplier_membrane.size(); i++)
            {
                Eigen::Vector3d constraint = Eigen::Vector3d::Zero();
                diffxpbd::Mat39 constraint_derivatives = diffxpbd::Mat39::Zero();
                diffxpbd::Mat99 constraint_mass_matrix = diffxpbd::Mat99::Zero();
                calMembraneConstraint_i(s.position_predicted_during_iteration, m, s, i,
                                        constraint, constraint_derivatives, constraint_mass_matrix);

                const diffxpbd::Mat33 alpha_tilde = s.compliance_membrane[i] / dt2;
                diffxpbd::Vec9d inv_mass = constraint_mass_matrix.diagonal().cwiseInverse();
                maskPinnedInverseMasses(inv_mass, m.indices_membrane[i]);
                const auto& J = constraint_derivatives;
                const diffxpbd::Mat33 lhs = J * inv_mass.asDiagonal() * J.transpose() + alpha_tilde;
                const Eigen::Vector3d rhs =
                    -constraint - alpha_tilde * s.multiplier_membrane[i];
                const Eigen::LDLT<diffxpbd::Mat33> factorization(lhs);
                if (factorization.info() != Eigen::Success ||
                    (factorization.vectorD().array() <= 0.0).any()) {
                    return false;
                }
                const Eigen::Vector3d delta_multiplier = factorization.solve(rhs);
                const diffxpbd::Vec9d delta_x =
                    inv_mass.cwiseProduct(J.transpose() * delta_multiplier);
                s.multiplier_membrane[i] += delta_multiplier;
                
                const auto [a, b, c] = m.triangles[i].indices;
                s.position_predicted_during_iteration[a] += delta_x.segment<3>(0);
                s.position_predicted_during_iteration[b] += delta_x.segment<3>(3);
                s.position_predicted_during_iteration[c] += delta_x.segment<3>(6);
            }

            // compute bending constraint
            for (int i = 0; i < s.multiplier_bending.size(); i++)
            {
                double constraint = 0.0;
                diffxpbd::Mat112 constraint_derivatives = diffxpbd::Mat112::Zero();
                diffxpbd::Mat1212 constraint_mass_matrix = diffxpbd::Mat1212::Zero();
                calBendingConstraint_i(s.position_predicted_during_iteration, m, s, i,
                    constraint, constraint_derivatives, constraint_mass_matrix);

                const double alpha_tilde = s.compliance_bending[i] / dt2;
                diffxpbd::Vecd<12> inv_mass = constraint_mass_matrix.diagonal().cwiseInverse();
                maskPinnedInverseMasses(inv_mass, m.indices_bending[i]);
                const auto& J = constraint_derivatives;
                const double denominator =
                    (J * inv_mass.asDiagonal() * J.transpose())(0, 0) + alpha_tilde;
                if (!std::isfinite(denominator) || denominator <= 0.0) {
                    return false;
                }
                const double delta_multiplier =
                    (-constraint - alpha_tilde * s.multiplier_bending[i]) / denominator;
                const diffxpbd::Vecd<12> delta_x =
                    inv_mass.cwiseProduct(J.transpose() * delta_multiplier);
                s.multiplier_bending[i] += delta_multiplier;
                const auto& hinge = m.bending_hinges[i];
                s.position_predicted_during_iteration[hinge.edge[0]] += delta_x.segment<3>(0);
                s.position_predicted_during_iteration[hinge.edge[1]] += delta_x.segment<3>(3);
                s.position_predicted_during_iteration[hinge.opposite[0]] += delta_x.segment<3>(6);
                s.position_predicted_during_iteration[hinge.opposite[1]] += delta_x.segment<3>(9);
            }

            // Refresh and project cloth self-contacts
            if (!computeContactConstraint()) {
                return false;
            }
            iteration++;
        };

    }
    else  // Jacobi style projection
    {
        int iteration = 0;
        while (iteration < params.totalIterations)
        {
            // compute membrane constraint
            for (int i = 0; i < s.multiplier_membrane.size(); i++)
            {
                Eigen::Vector3d constraint = Eigen::Vector3d::Zero();
                diffxpbd::Mat39 constraint_derivatives = diffxpbd::Mat39::Zero();
                diffxpbd::Mat99 constraint_mass_matrix = diffxpbd::Mat99::Zero();
                calMembraneConstraint_i(s.position_predicted_after_iteration, m, s, i,
                    constraint, constraint_derivatives, constraint_mass_matrix);

                const diffxpbd::Mat33 alpha_tilde = s.compliance_membrane[i] / dt2;
                diffxpbd::Vec9d inv_mass = constraint_mass_matrix.diagonal().cwiseInverse();
                maskPinnedInverseMasses(inv_mass, m.indices_membrane[i]);
                const auto& J = constraint_derivatives;
                const diffxpbd::Mat33 lhs = J * inv_mass.asDiagonal() * J.transpose() + alpha_tilde;
                const Eigen::Vector3d rhs =
                    -constraint - alpha_tilde * s.multiplier_membrane[i];
                const Eigen::LDLT<diffxpbd::Mat33> factorization(lhs);
                if (factorization.info() != Eigen::Success ||
                    (factorization.vectorD().array() <= 0.0).any()) {
                    return false;
                }
                const Eigen::Vector3d delta_multiplier = factorization.solve(rhs);
                const diffxpbd::Vec9d delta_x =
                    inv_mass.cwiseProduct(J.transpose() * delta_multiplier);
                s.multiplier_membrane[i] += delta_multiplier;
                const auto [a, b, c] = m.triangles[i].indices;
                s.position_predicted_during_iteration[a] += delta_x.segment<3>(0);
                s.position_predicted_during_iteration[b] += delta_x.segment<3>(3);
                s.position_predicted_during_iteration[c] += delta_x.segment<3>(6);
            }

            // compute bending constraint
            for (int i = 0; i < s.multiplier_bending.size(); i++)
            {
                double constraint = 0.0;
                diffxpbd::Mat112 constraint_derivatives = diffxpbd::Mat112::Zero();
                diffxpbd::Mat1212 constraint_mass_matrix = diffxpbd::Mat1212::Zero();
                calBendingConstraint_i(s.position_predicted_after_iteration, m, s, i,
                    constraint, constraint_derivatives, constraint_mass_matrix);

                const double alpha_tilde = s.compliance_bending[i] / dt2;
                diffxpbd::Vecd<12> inv_mass = constraint_mass_matrix.diagonal().cwiseInverse();
                maskPinnedInverseMasses(inv_mass, m.indices_bending[i]);
                const auto& J = constraint_derivatives;
                const double denominator =
                    (J * inv_mass.asDiagonal() * J.transpose())(0, 0) + alpha_tilde;
                if (!std::isfinite(denominator) || denominator <= 0.0) {
                    return false;
                }
                const double delta_multiplier =
                    (-constraint - alpha_tilde * s.multiplier_bending[i]) / denominator;
                const diffxpbd::Vecd<12> delta_x =
                    inv_mass.cwiseProduct(J.transpose() * delta_multiplier);
                s.multiplier_bending[i] += delta_multiplier;
                const auto& hinge = m.bending_hinges[i];
                s.position_predicted_during_iteration[hinge.edge[0]] += delta_x.segment<3>(0);
                s.position_predicted_during_iteration[hinge.edge[1]] += delta_x.segment<3>(3);
                s.position_predicted_during_iteration[hinge.opposite[0]] += delta_x.segment<3>(6);
                s.position_predicted_during_iteration[hinge.opposite[1]] += delta_x.segment<3>(9);
            }

            // Refresh and project cloth self-contacts
            if (!computeContactConstraint()) {
                return false;
            }
			s.position_predicted_after_iteration = s.position_predicted_during_iteration;
			iteration++;
        };

    }


    if (params.contact.self_collision) {
        for (int pass = 0; pass < params.contact.max_extra_contact_sweeps; ++pass) {
            const auto count = s.contacts.size();
            if (!computeContactConstraint()) return false;
            if (s.contacts.size() == count) break;
        }
        s.position_predicted_after_iteration = s.position_predicted_during_iteration;

        // Compliant separation can be slightly violated, but an actual crossing is rejected.
        auto crossing_settings = params.contact;
        crossing_settings.separation = 0.0;
        const auto crossings = detectContacts(crossing_settings,
            static_cast<const std::vector<diffxpbd::Contact>*>(nullptr));
        for (const auto& contact : crossings) {
            const double toi = contact.time_of_impact;
            const double final_gap = diffxpbd::evaluateContact(contact,
                s.position_predicted_during_iteration, 0.0).gap;
            if (toi > 1e-8 || final_gap < -params.contact.ccd_tolerance) return false;
        }
    }

    const auto& projected_positions = params.JacobiSolver
        ? s.position_predicted_after_iteration
        : s.position_predicted_during_iteration;
    for (const auto& position : projected_positions)
        if (!position.allFinite()) return false;
    for (std::size_t i = 0; i < s.velocity.size(); ++i) {
        if (s.pinned_positions[i]) {
            s.velocity[i].setZero();
        } else {
            s.velocity[i] = (projected_positions[i] - s.position[i]) / params.timestep;
        }
    }
    s.position = projected_positions;

    return true;
}

} // namespace

bool advanceState(const mesh& m, const material& mat, const parameters& params, state& s)
{
    const auto& settings = params.contact;
    if (!(params.timestep > 0.0) || params.totalIterations < 1 ||
        settings.max_extra_contact_step_levels < 0 ||
        settings.max_extra_contact_step_levels > 10) return false;
    if (!settings.self_collision) return advanceStateOnce(m, mat, params, s);

    for (int level = 0; level <= settings.max_extra_contact_step_levels; ++level) {
        const int count = 1 << level;
        state trial = s;
        parameters substep_params = params;
        substep_params.timestep = params.timestep / count;
        bool success = true;
        for (int substep = 0; substep < count; ++substep) {
            try {
                success = advanceStateOnce(m, mat, substep_params, trial);
            } catch (const std::runtime_error&) {
                success = false;
            }
            if (!success) break;
        }
        if (success) {
            trial.contact_substeps = count;
            s = std::move(trial);
            return true;
        }
    }
    return false;
}
