#include "diffxpbd/kinematic_collider.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace diffxpbd {

KinematicCollider::KinematicCollider(std::size_t vertex_count,
                                     std::vector<Triangle> triangles,
                                     PositionSampler motion)
    : vertex_count_(vertex_count), triangles_(std::move(triangles)),
      motion_(std::move(motion))
{
    if (vertex_count_ < 3 || triangles_.empty() || !motion_) {
        throw std::invalid_argument("A kinematic collider needs surface triangles and a motion sampler");
    }
    for (const auto& [a, b, c] : triangles_) {
        if (a >= vertex_count_ || b >= vertex_count_ || c >= vertex_count_ ||
            a == b || a == c || b == c) {
            throw std::invalid_argument("Invalid kinematic collider triangle indices");
        }
    }
}

const std::vector<KinematicCollider::Triangle>& KinematicCollider::triangles() const noexcept
{
    return triangles_;
}

std::vector<Eigen::Vector3d> KinematicCollider::sample(double time) const
{
    if (!std::isfinite(time)) {
        throw std::invalid_argument("Kinematic collider sample time must be finite");
    }

    auto positions = motion_(time);
    if (positions.size() != vertex_count_) {
        throw std::runtime_error("Kinematic collider motion changed the vertex count");
    }
    for (const auto& position : positions) {
        if (!position.allFinite()) {
            throw std::runtime_error("Kinematic collider motion returned a nonfinite position");
        }
    }
    return positions;
}

ColliderStep KinematicCollider::sampleStep(double start_time, double end_time) const
{
    if (!std::isfinite(start_time) || !std::isfinite(end_time) || end_time <= start_time) {
        throw std::invalid_argument("A collider step needs finite, increasing endpoint times");
    }
    return {start_time, end_time, sample(start_time), sample(end_time)};
}

} // namespace diffxpbd
