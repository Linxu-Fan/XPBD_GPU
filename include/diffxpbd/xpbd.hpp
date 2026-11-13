#pragma once
#include "diffxpbd/contact.hpp"
#include "diffxpbd/kinematic_collider.hpp"
#include <array>
#include <cstddef>
#include <iostream>
#include <optional>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <filesystem>

namespace diffxpbd {

	template <int Rows, int Cols>
	using Mat = Eigen::Matrix<double, Rows, Cols, Eigen::RowMajor>;
	using Mat22 = Eigen::Matrix2d;
	using Mat33 = Eigen::Matrix3d;
	using Mat32 = Mat<3, 2>;
	using Mat39 = Mat<3, 9>;
	using Mat99 = Mat<9, 9>;
	using Mat112 = Mat<1, 12>;
	using Mat1212 = Mat<12, 12>;


	template <int Rows>
	using Vecd = Eigen::Matrix<double, Rows, 1>;
	using Vec9d = Vecd<9>;

	template <int Rows>
	using Veci = Eigen::Matrix<int, Rows, 1>;
	using Vec9i = Veci<9>;
	using Vec12i = Veci<12>;
	

} // namespace diffxpbd


// XPBD mesh 
struct mesh_triangle
{
    std::array<std::size_t, 3> indices;
    Eigen::Vector3d warp;
    Eigen::Vector3d weft;
};

struct bending_hinge
{
    std::array<std::size_t, 2> edge;
    std::array<std::size_t, 2> opposite;
    double rest_angle; // signed dihedral angle in radians
};

// Cloth mesh struct but could also be used for other triangle meshes
struct mesh 
{
	std::vector<Eigen::Vector3d> vertices;
	std::vector<Eigen::Vector3d> velocities;
	std::vector<mesh_triangle> triangles;
	// Rest topology caches populated by initializeState; T triangles and H hinges.
	std::vector<diffxpbd::Vec9i> indices_membrane; // T entries, each 9x1; global x/y/z DOFs for three vertices
	std::vector<diffxpbd::Vec12i> indices_bending; // H entries, each 12x1; global DOFs for edge, then opposite vertices
	std::vector<bending_hinge> bending_hinges; // H entries; each has two edge indices, two opposite indices, and one rest angle
};

// simulation parameters
struct parameters
{
	double timestep = 0.01;
	int totalSteps = 4000; // total simulation timesteps
	int totalIterations = 50; // maximum simulation iterations
	Eigen::Vector3d gravity = {0.0, -9.8, 0.0};
	bool JacobiSolver = false;
    diffxpbd::ContactSettings contact;
};

// material properties for simulation
struct material
{
	// general 3D material parameters
	double density = 0.22658 / 0.00052; // kg/m^3, bulk density from measured areal mass and thickness
	double youngsModulus = 1e5; // Pa
	double poissonRatio = 0.3; // dimensionless

	// Linearized woven cotton-blend twill preset (M1, doi:10.3390/polym17152013).
	double C00 = 482.0; // N/m, warp
	double C11 = 441.0; // N/m, weft
	double C01 = 0.0;   // N/m, uncoupled approximation
	double C22 = 100.0; // N/m, engineering shear
	double b = 3.9e-5; // N m, mean measured rigidity scaled by sqrt(3) for a regular hinge
	double thickness = 0.00052; // m
};

// N = cloth vertex count, 
// T = triangle count, 
// H = interior bending hinge count,
// K = contact count in the last internal substep. 
struct state
{
	std::vector<diffxpbd::Mat22> Dm_inv; // T entries, each 2x2; inverse rest edge matrix
	
	Eigen::MatrixXd M; // 3N x 3N; diagonal lumped mass matrix, x/y/z per vertex

	std::vector<Eigen::Vector3d> f_ext; // N entries, each 3x1; external force per vertex

	std::vector<Eigen::Vector3d> position; // N entries, each 3x1; current world-space position
	std::vector<Eigen::Vector3d> velocity; // N entries, each 3x1; current world-space velocity
	std::vector<Eigen::Vector3d> position_predicted; // N entries, each 3x1; prediction before constraints
	std::vector<std::optional<Eigen::Vector3d>> pinned_positions; // N entries; optional 3x1 anchor, absent for free vertices
	std::vector<Eigen::Vector3d> position_predicted_during_iteration; // N entries, each 3x1; updated during constraint sweeps
	std::vector<Eigen::Vector3d> position_predicted_after_iteration; // N entries, each 3x1; positions after a constraint sweep

	std::vector<Eigen::Vector3d> multiplier_membrane; // T entries, each 3x1; membrane multipliers.  
	std::vector<Eigen::Vector3d> constraint_membrane; // T entries, each 3x1; warp, weft, engineering shear strains. The constraint is denoted using Voigt notation.
	std::vector<diffxpbd::Mat33> compliance_membrane; // T entries, each 3x3; unscaled compliance

	std::vector<double> multiplier_bending; // H scalar entries; bending multipliers
	std::vector<double> constraint_bending; // H scalar entries; dihedral angle changes
	std::vector<double> compliance_bending; // H scalar entries; unscaled compliance

    std::vector<diffxpbd::Contact> contacts; // K Contact records; active feature geometry of the last internal substep
    std::vector<double> multiplier_contact; // K scalar entries; nonnegative contact multipliers
    int contact_substeps = 1; // one integer; accepted internal substep count, for replay

};


// Import triangle OBJ geometry, zero velocities, and synthetic rest directions.
// Warp follows projected -Y (projected +X for nearly horizontal triangles).
// Weft is the face normal crossed with warp.
// Throws std::runtime_error on unreadable files or invalid geometry.
void importAnMesh(const std::filesystem::path& mesh_path, mesh& m);

// Write current simulated positions and the original triangle topology as an OBJ.
// Append the timestep index before the extension (skirt.obj becomes skirt_42.obj).
// Throws std::runtime_error on invalid geometry or output failure.
void exportStateAsObj(const std::filesystem::path& mesh_path, const mesh& m, const state& s,
                      std::size_t timestep);

// Export prescribed geometry using the same timestep naming and validation.
void exportStateAsObj(const std::filesystem::path& mesh_path, const mesh& m,
                      const std::vector<Eigen::Vector3d>& positions, std::size_t timestep);


// Initialize mesh topology caches, state rest data, lumped masses, and unscaled compliances.
// Returns false for invalid geometry or invalid material parameters.
bool initializeState(mesh& m, const material& mat, state& s);

// Replace f_ext with one gravity force (mass times acceleration) per vertex.
// Returns false if the mass matrix or gravity is invalid.
bool calExternalForce(const mesh& m, const parameters& params, state& s);


// Evaluate one triangle's strain and its 3x9 position Jacobian.
// Jacobian columns and the local lumped mass matrix are ordered x/y/z per vertex.
// Uses the supplied full mesh position buffer, mesh topology, and cached rest data.
void calMembraneConstraint_i(const std::vector<Eigen::Vector3d>& positions, const mesh& m, const state& s,
                             int index, Eigen::Vector3d& constraint,
                             diffxpbd::Mat39& constraint_derivatives,
                             diffxpbd::Mat99& constraint_mass_matrix);


// Evaluate one hinge's wrapped dihedral change and its 1x12 position Jacobian.
// Columns and local lumped masses are ordered x/y/z for the two edge vertices, then the
// two opposite vertices. Uses the supplied full mesh position buffer.
// The Jacobian follows the local angle branch.
void calBendingConstraint_i(const std::vector<Eigen::Vector3d>& positions, const mesh& m, const state& s,
                            int index, double& constraint,
                            diffxpbd::Mat112& constraint_derivatives,
                            diffxpbd::Mat1212& constraint_mass_matrix);


// one step forward simulation of the mesh state using XPBD, with optional kinematic collider.
// Collider input is borrowed for this call and remains outside the dynamic state.
// Returns false if the supplied collider geometry or substep duration is invalid.
// Mesh contact is frictionless and preserves pins. Endpoint motion is piecewise linear.
// Collider calls commit state only after successful contact validation and bounded retries.
bool advanceState(const mesh& m, const material& mat, const parameters& params, state& s,
                  const diffxpbd::ColliderInput* collider = nullptr);


