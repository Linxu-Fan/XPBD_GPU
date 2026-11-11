#pragma once

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <optional>

namespace diffxpbd {

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

} // namespace diffxpbd
