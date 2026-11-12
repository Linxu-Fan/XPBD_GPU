#include "diffxpbd/contact.hpp"
#include "diffxpbd/xpbd.hpp"

#include <tight_inclusion/ccd.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace diffxpbd {
namespace {

using Vec3 = Eigen::Vector3d;
using Positions = std::vector<Vec3>;
using PairKey = std::tuple<int, std::array<std::size_t, 3>, int, std::array<std::size_t, 3>>;

Vec3 contactPoint(const ContactPoint& point, const Positions& positions)
{
    Vec3 result = Vec3::Zero();
    for (std::size_t i = 0; i < point.vertex_count(); ++i) {
        result += point.weights[i] * positions[point.indices[i]];
    }
    return result;
}

struct Box
{
    Vec3 minimum = Vec3::Constant(std::numeric_limits<double>::infinity());
    Vec3 maximum = -minimum;

    void include(const Vec3& point)
    {
        minimum = minimum.cwiseMin(point);
        maximum = maximum.cwiseMax(point);
    }

    bool overlaps(const Box& other, double margin) const
    {
        return (minimum.array() <= other.maximum.array() + margin).all() &&
               (other.minimum.array() <= maximum.array() + margin).all();
    }
};

struct Feature
{
    ContactFeatureType type;
    std::array<std::size_t, 3> indices{};

    std::size_t size() const { return static_cast<std::size_t>(type); }
};

struct Features
{
    std::vector<Feature> vertices;
    std::vector<Feature> faces;
    std::vector<Feature> edges;
};

Features makeFeatures(const std::vector<KinematicCollider::Triangle>& triangles)
{
    Features result;
    std::set<std::size_t> vertices;
    std::set<std::array<std::size_t, 2>> edges;
    for (const auto& triangle : triangles) {
        result.faces.push_back({ContactFeatureType::Triangle, triangle});
        for (std::size_t i = 0; i < 3; ++i) {
            vertices.insert(triangle[i]);
            const auto a = triangle[i];
            const auto b = triangle[(i + 1) % 3];
            edges.insert({std::min(a, b), std::max(a, b)});
        }
    }
    for (const auto vertex : vertices) {
        result.vertices.push_back({ContactFeatureType::Vertex, {vertex, 0, 0}});
    }
    for (const auto& edge : edges) {
        result.edges.push_back({ContactFeatureType::Edge, {edge[0], edge[1], 0}});
    }
    return result;
}

Box sweptBox(const Feature& feature, const Positions& start, const Positions& end)
{
    Box box;
    for (std::size_t i = 0; i < feature.size(); ++i) {
        box.include(start[feature.indices[i]]);
        box.include(end[feature.indices[i]]);
    }
    return box;
}

// Median-split swept AABB tree. Candidate order is sorted for reproducible solves.
class BoxTree
{
    struct Node
    {
        Box box;
        std::size_t begin;
        std::size_t end;
        int left = -1;
        int right = -1;
    };
    std::vector<Box> boxes_;
    std::vector<std::size_t> order_;
    std::vector<Node> nodes_;

    int build(std::size_t begin, std::size_t end)
    {
        Box bounds;
        for (auto i = begin; i < end; ++i) {
            bounds.include(boxes_[order_[i]].minimum);
            bounds.include(boxes_[order_[i]].maximum);
        }
        const int node = static_cast<int>(nodes_.size());
        nodes_.push_back({bounds, begin, end});
        if (end - begin > 8) {
            Eigen::Index axis;
            (bounds.maximum - bounds.minimum).maxCoeff(&axis);
            const auto middle = begin + (end - begin) / 2;
            std::nth_element(order_.begin() + begin, order_.begin() + middle,
                             order_.begin() + end, [&](auto a, auto b) {
                const double ca = boxes_[a].minimum[axis] + boxes_[a].maximum[axis];
                const double cb = boxes_[b].minimum[axis] + boxes_[b].maximum[axis];
                return ca == cb ? a < b : ca < cb;
            });
            const int left = build(begin, middle);
            const int right = build(middle, end);
            nodes_[node].left = left;
            nodes_[node].right = right;
        }
        return node;
    }

    void query(int node, const Box& box, double margin, std::vector<std::size_t>& result) const
    {
        const auto& entry = nodes_[node];
        if (!entry.box.overlaps(box, margin)) return;
        if (entry.left >= 0) {
            query(entry.left, box, margin, result);
            query(entry.right, box, margin, result);
        } else {
            for (auto i = entry.begin; i < entry.end; ++i) {
                if (boxes_[order_[i]].overlaps(box, margin)) result.push_back(order_[i]);
            }
        }
    }

public:
    BoxTree(const std::vector<Feature>& features, const Positions& start, const Positions& end)
    {
        for (const auto& feature : features) boxes_.push_back(sweptBox(feature, start, end));
        order_.resize(boxes_.size());
        std::iota(order_.begin(), order_.end(), 0);
        if (!order_.empty()) build(0, order_.size());
    }

    std::vector<std::size_t> candidates(const Box& box, double margin) const
    {
        std::vector<std::size_t> result;
        if (!nodes_.empty()) query(0, box, margin, result);
        std::sort(result.begin(), result.end());
        return result;
    }
};

double closestSegmentParameter(const Vec3& point, const Vec3& a, const Vec3& b)
{
    const Vec3 edge = b - a;
    const double length2 = edge.squaredNorm();
    return length2 > 1e-30 ? std::clamp(edge.dot(point - a) / length2, 0.0, 1.0) : 0.0;
}

std::array<double, 3> triangleWeights(const Vec3& point, const Vec3& a,
                                     const Vec3& b, const Vec3& c)
{
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = point - a;
    const double denominator = ab.squaredNorm() * ac.squaredNorm() - std::pow(ab.dot(ac), 2);
    if (denominator > 1e-30) {
        const double v = (ac.squaredNorm() * ab.dot(ap) - ab.dot(ac) * ac.dot(ap)) / denominator;
        const double w = (ab.squaredNorm() * ac.dot(ap) - ab.dot(ac) * ab.dot(ap)) / denominator;
        if (v >= 0.0 && w >= 0.0 && v + w <= 1.0) return {1.0 - v - w, v, w};
    }
    const std::array<Vec3, 3> vertices{a, b, c};
    std::array<double, 3> best{};
    double distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 3; ++i) {
        const int j = (i + 1) % 3;
        const double t = closestSegmentParameter(point, vertices[i], vertices[j]);
        const double candidate = (point - ((1.0 - t) * vertices[i] + t * vertices[j])).squaredNorm();
        if (candidate < distance) {
            distance = candidate;
            best = {};
            best[i] = 1.0 - t;
            best[j] = t;
        }
    }
    return best;
}

std::array<double, 2> edgeParameters(const Vec3& a, const Vec3& b,
                                    const Vec3& c, const Vec3& d)
{
    const Vec3 u = b - a;
    const Vec3 v = d - c;
    const Vec3 r = a - c;
    const double uu = u.squaredNorm();
    const double vv = v.squaredNorm();
    if (uu <= 1e-30) return {0.0, closestSegmentParameter(a, c, d)};
    if (vv <= 1e-30) return {closestSegmentParameter(c, a, b), 0.0};
    const double uv = u.dot(v);
    const double ur = u.dot(r);
    const double vr = v.dot(r);
    const double denominator = uu * vv - uv * uv;
    double s = denominator > 1e-30
        ? std::clamp((uv * vr - ur * vv) / denominator, 0.0, 1.0) : 0.0;
    double t = (uv * s + vr) / vv;
    if (t < 0.0) {
        t = 0.0;
        s = std::clamp(-ur / uu, 0.0, 1.0);
    } else if (t > 1.0) {
        t = 1.0;
        s = std::clamp((uv - ur) / uu, 0.0, 1.0);
    }
    return {s, t};
}

Contact makeContact(const Feature& first, const Feature& second,
                    const std::array<Vec3, 4>& hit, const std::array<Vec3, 4>& before,
                    bool edge_edge, bool reverse, double toi)
{
    Contact contact;
    ContactPoint a{first.type, first.indices, {}};
    ContactPoint b{second.type, second.indices, {}};
    Vec3 delta, prior_delta, fallback;
    if (edge_edge) {
        const auto parameters = edgeParameters(hit[0], hit[1], hit[2], hit[3]);
        a.weights = {1.0 - parameters[0], parameters[0], 0.0};
        b.weights = {1.0 - parameters[1], parameters[1], 0.0};
        delta = a.weights[0] * hit[0] + a.weights[1] * hit[1] -
                b.weights[0] * hit[2] - b.weights[1] * hit[3];
        prior_delta = a.weights[0] * before[0] + a.weights[1] * before[1] -
                      b.weights[0] * before[2] - b.weights[1] * before[3];
        fallback = (hit[1] - hit[0]).cross(hit[3] - hit[2]);
    } else {
        a.weights = {1.0, 0.0, 0.0};
        b.weights = triangleWeights(hit[0], hit[1], hit[2], hit[3]);
        delta = hit[0] - b.weights[0] * hit[1] - b.weights[1] * hit[2] - b.weights[2] * hit[3];
        prior_delta = before[0] - b.weights[0] * before[1] -
                      b.weights[1] * before[2] - b.weights[2] * before[3];
        fallback = (hit[2] - hit[1]).cross(hit[3] - hit[1]);
    }
    Vec3 normal = delta;
    if (normal.norm() < 1e-12) {
        normal = fallback.norm() > 1e-12 ? fallback : prior_delta;
        if (normal.norm() < 1e-12) {
            throw std::runtime_error("Cannot determine a contact normal for coincident degenerate features");
        }
        if (normal.dot(prior_delta) < 0.0) normal = -normal;
    }
    contact.cloth = reverse ? b : a;
    contact.obstacle = reverse ? a : b;
    contact.normal = (reverse ? -normal : normal).normalized();
    contact.time_of_impact = toi;
    return contact;
}

void detectPairs(const std::vector<Feature>& first, const Positions& a0, const Positions& a1,
                 const std::vector<Feature>& second, const Positions& b0, const Positions& b1,
                 bool edge_edge, bool reverse, const ContactSettings& settings,
                 const std::set<PairKey>& known, std::vector<Contact>& contacts)
{
    const BoxTree tree(second, b0, b1);
    const Eigen::Array3d error = Eigen::Array3d::Constant(-1.0);
    for (const auto& a : first) {
        for (const auto candidate : tree.candidates(sweptBox(a, a0, a1),
                                                    settings.separation + settings.ccd_tolerance)) {
            const auto& b = second[candidate];
            const auto& cloth = reverse ? b : a;
            const auto& obstacle = reverse ? a : b;
            if (known.count({static_cast<int>(cloth.type), cloth.indices,
                             static_cast<int>(obstacle.type), obstacle.indices})) continue;
            std::array<Vec3, 4> start, end;
            if (edge_edge) {
                start = {a0[a.indices[0]], a0[a.indices[1]], b0[b.indices[0]], b0[b.indices[1]]};
                end = {a1[a.indices[0]], a1[a.indices[1]], b1[b.indices[0]], b1[b.indices[1]]};
            } else {
                start = {a0[a.indices[0]], b0[b.indices[0]], b0[b.indices[1]], b0[b.indices[2]]};
                end = {a1[a.indices[0]], b1[b.indices[0]], b1[b.indices[1]], b1[b.indices[2]]};
            }
            // Existing proximity is a contact at time zero, so it needs no root search.
            std::array<Vec3, 4> prior;
            for (int i = 0; i < 4; ++i) prior[i] = start[i] - 1e-4 * (end[i] - start[i]);
            auto initial = makeContact(a, b, start, prior, edge_edge, reverse, 0.0);
            const double initial_gap = reverse
                ? evaluateContact(initial, b0, a0, settings.separation).gap
                : evaluateContact(initial, a0, b0, settings.separation).gap;
            if (initial_gap <= settings.ccd_tolerance) {
                contacts.push_back(std::move(initial));
                continue;
            }
            double toi = 0.0;
            double output_tolerance = 0.0;
            const bool hit = edge_edge
                ? ticcd::edgeEdgeCCD(start[0], start[1], start[2], start[3],
                    end[0], end[1], end[2], end[3], error, settings.separation,
                    toi, settings.ccd_tolerance, 1.0, 100000, output_tolerance, false,
                    ticcd::CCDRootFindingMethod::BREADTH_FIRST_SEARCH)
                : ticcd::vertexFaceCCD(start[0], start[1], start[2], start[3],
                    end[0], end[1], end[2], end[3], error, settings.separation,
                    toi, settings.ccd_tolerance, 1.0, 100000, output_tolerance, false,
                    ticcd::CCDRootFindingMethod::BREADTH_FIRST_SEARCH);
            if (!hit) continue;
            if (!std::isfinite(toi) || toi < 0.0 || toi > 1.0 ||
                !std::isfinite(output_tolerance)) {
                throw std::runtime_error("CCD returned an invalid time of impact");
            }
            std::array<Vec3, 4> impact, before;
            const double previous_time = std::max(0.0, toi - 1e-4);
            for (int i = 0; i < 4; ++i) {
                impact[i] = (1.0 - toi) * start[i] + toi * end[i];
                before[i] = (1.0 - previous_time) * start[i] + previous_time * end[i];
            }
            contacts.push_back(makeContact(a, b, impact, before, edge_edge, reverse, toi));
        }
    }
}

} // namespace

ContactEvaluation evaluateContact(const Contact& contact, const Positions& cloth_positions,
                                   const Positions& obstacle_positions, double separation)
{
    ContactEvaluation result;
    result.gap = contact.normal.dot(contactPoint(contact.cloth, cloth_positions) -
                                    contactPoint(contact.obstacle, obstacle_positions)) - separation;
    for (std::size_t i = 0; i < contact.cloth.vertex_count(); ++i)
        result.cloth_jacobian.row(i) = contact.cloth.weights[i] * contact.normal.transpose();
    for (std::size_t i = 0; i < contact.obstacle.vertex_count(); ++i)
        result.obstacle_jacobian.row(i) = -contact.obstacle.weights[i] * contact.normal.transpose();
    return result;
}

std::vector<Contact> detectMeshContacts(const mesh& cloth, const Positions& start_positions,
    const Positions& end_positions, const ColliderInput& collider, const ContactSettings& settings,
    const std::vector<Contact>* known_contacts)
{
    std::set<PairKey> known;
    if (known_contacts) {
        for (const auto& contact : *known_contacts)
            known.emplace(static_cast<int>(contact.cloth.feature_type), contact.cloth.indices,
                          static_cast<int>(contact.obstacle.feature_type), contact.obstacle.indices);
    }
    std::vector<KinematicCollider::Triangle> triangles;
    for (const auto& triangle : cloth.triangles) triangles.push_back(triangle.indices);
    const auto cloth_features = makeFeatures(triangles);
    const auto obstacle_features = makeFeatures(collider.triangles);
    const auto& step = collider.step;
    std::vector<Contact> contacts;
    detectPairs(cloth_features.vertices, start_positions, end_positions,
                obstacle_features.faces, step.start_positions, step.end_positions,
                false, false, settings, known, contacts);
    detectPairs(obstacle_features.vertices, step.start_positions, step.end_positions,
                cloth_features.faces, start_positions, end_positions,
                false, true, settings, known, contacts);
    detectPairs(cloth_features.edges, start_positions, end_positions,
                obstacle_features.edges, step.start_positions, step.end_positions,
                true, false, settings, known, contacts);
    return contacts;
}

} // namespace diffxpbd
