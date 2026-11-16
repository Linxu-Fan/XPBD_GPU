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
    // Build the median partition once. Subsequent queries preserve that tree
    // topology and update every leaf and ancestor bound. Bounds always enclose
    // BOTH motion endpoints, including position corrections made by the solver.
    // Refit changes traversal cost, but cannot remove an overlapping primitive.
    void refit(const std::vector<Feature>& features, const Positions& start, const Positions& end)
    {
        const bool rebuild = boxes_.size() != features.size();
        boxes_.resize(features.size());
        for (std::size_t i = 0; i < features.size(); ++i)
            boxes_[i] = sweptBox(features[i], start, end);
        if (rebuild || (nodes_.empty() && !features.empty())) {
            nodes_.clear();
            order_.resize(boxes_.size());
            std::iota(order_.begin(), order_.end(), 0);
            if (!order_.empty()) build(0, order_.size());
            return;
        }
        // build() appends children after parents. Visiting nodes backwards makes
        // each child's updated bound available before merging its parent.
        for (std::size_t i = nodes_.size(); i-- > 0;) {
            auto& node = nodes_[i];
            Box bounds;
            if (node.left >= 0) {
                for (const auto child : {node.left, node.right}) {
                    bounds.include(nodes_[child].box.minimum);
                    bounds.include(nodes_[child].box.maximum);
                }
            } else {
                for (auto j = node.begin; j < node.end; ++j) {
                    bounds.include(boxes_[order_[j]].minimum);
                    bounds.include(boxes_[order_[j]].maximum);
                }
            }
            node.box = bounds;
        }
    }

    void clear()
    {
        boxes_.clear();
        order_.clear();
        nodes_.clear();
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

// Each candidate retains a possible separating direction, not a cached CCD
// hit/miss. We recheck this direction against ALL current endpoints before
// skipping a root search. A stale direction can only lose speed, not safety.
struct Candidate
{
    std::size_t index;
    Vec3 separating_direction = Vec3::Zero();
};

struct PairCache
{
    Positions a0, a1, b0, b1;
    bool first_query = true;
    double margin = -1.0;
    double padding = 0.0;
    std::vector<std::vector<Candidate>> candidates;

    bool covers(const Positions& previous, const Positions& current) const
    {
        if (previous.size() != current.size()) return false;
        for (std::size_t i = 0; i < current.size(); ++i)
            if (!((current[i] - previous[i]).array().abs() <= padding).all()) return false;
        return true;
    }
};

// A fixed plane separates two moving convex features for the entire substep if
// their projected endpoint intervals are disjoint. Every interpolated point is
// a convex combination of these endpoints. Thus this is a sufficient condition
// for absence of contact, including contacts between endpoints (tunneling).
// It is deliberately one-sided: failure to certify always falls back to CCD.
bool separatedOverStep(const Vec3& normal, const std::array<Vec3, 4>& start,
                       const std::array<Vec3, 4>& end, bool edge_edge,
                       const ContactSettings& settings)
{
    if (normal.isZero(0.0)) return false;
    double first_min = std::numeric_limits<double>::infinity();
    double second_max = -first_min;
    double scale = 1.0;
    for (int i = 0; i < 4; ++i) {
        for (const auto* positions : {&start, &end}) {
            const auto& point = (*positions)[i];
            scale = std::max(scale, point.cwiseAbs().maxCoeff());
            const double projection = normal.dot(point);
            if (i < (edge_edge ? 2 : 1)) first_min = std::min(first_min, projection);
            else second_max = std::max(second_max, projection);
        }
    }
    // Unit normals make the projected gap a distance in meters. The additional
    // cushion avoids accepting a certificate on a floating-point boundary.
    const double cushion = settings.ccd_tolerance +
        64.0 * std::numeric_limits<double>::epsilon() * scale;
    return first_min - second_max > settings.separation + cushion;
}

// A single plane can fail to separate full swept hulls even when features
// never touch: different vertices can occupy the same region at different times.
// Certify smaller time intervals instead. Every accepted interval has a plane
// separating ALL of its endpoint vertices, so this is a continuous-time proof,
// not a sampled-distance collision test. Bounds on work/depth only cause fallback
// to the original CCD query; exhausting them never implies a safe trajectory.
bool certifySeparatedIntervals(const std::array<Vec3, 4>& start,
                                const std::array<Vec3, 4>& end, bool edge_edge,
                                const ContactSettings& settings)
{
    struct Interval { double begin, end; int depth; };
    std::array<Interval, 16> pending;
    std::size_t count = 1;
    pending[0] = {0.0, 1.0, 0};
    for (int work = 0; count > 0 && work < 64; ++work) {
        const auto interval = pending[--count];
        std::array<Vec3, 4> lo, hi, mid;
        for (int i = 0; i < 4; ++i) {
            lo[i] = (1.0 - interval.begin) * start[i] + interval.begin * end[i];
            hi[i] = (1.0 - interval.end) * start[i] + interval.end * end[i];
            mid[i] = 0.5 * lo[i] + 0.5 * hi[i];
        }
        Vec3 delta;
        if (edge_edge) {
            const auto uv = edgeParameters(mid[0], mid[1], mid[2], mid[3]);
            delta = (1.0 - uv[0]) * mid[0] + uv[0] * mid[1] -
                    (1.0 - uv[1]) * mid[2] - uv[1] * mid[3];
        } else {
            const auto weights = triangleWeights(mid[0], mid[1], mid[2], mid[3]);
            delta = mid[0] - weights[0] * mid[1] - weights[1] * mid[2] - weights[2] * mid[3];
        }
        // Closest points only suggest a direction. Acceptance uses projection
        // bounds for ALL feature vertices; it never trusts a sampled distance
        // as a lower bound. Thus even a poorly conditioned closest-point solve
        // cannot produce an unsafe certificate.
        const double length = delta.norm();
        if (!(length > settings.separation + settings.ccd_tolerance)) return false;
        if (separatedOverStep(delta / length, lo, hi, edge_edge, settings)) continue;
        if (interval.depth >= 12) return false;
        const double middle = 0.5 * (interval.begin + interval.end);
        pending[count++] = {middle, interval.end, interval.depth + 1};
        pending[count++] = {interval.begin, middle, interval.depth + 1};
    }
    return count == 0;
}

void detectPairs(const std::vector<Feature>& first, const Positions& a0, const Positions& a1,
                 const std::vector<Feature>& second, const Positions& b0, const Positions& b1,
                 bool edge_edge, bool reverse, const ContactSettings& settings,
                 const std::set<PairKey>& known, std::vector<Contact>& contacts,
                 BoxTree& tree, PairCache& cache, bool validate_trajectory, bool self_contact = false)
{
    if (!std::isfinite(settings.collision_cache_padding) || settings.collision_cache_padding < 0.0)
        throw std::invalid_argument("Collision cache padding must be finite and nonnegative");
    const double margin = settings.separation + settings.ccd_tolerance;
    // Inflate BOTH feature boxes by padding. The equivalent query margin is
    // separation + 2*padding. If every endpoint stays inside its saved budget,
    // the saved list is a superset of every possible current broad-phase pair.
    // The exact current-box test below removes surplus pairs in the old order.
    const bool reuse = settings.cache_collision_queries && cache.margin >= margin &&
        cache.padding == settings.collision_cache_padding &&
        cache.covers(cache.a0, a0) && cache.covers(cache.a1, a1) &&
        cache.covers(cache.b0, b0) && cache.covers(cache.b1, b1);
    // Self-contact retains full swept discovery on every refresh: delaying
    // layer/edge events can change fold ordering. Proximity scheduling targets
    // the expensive kinematic-obstacle queries; both kinds reuse candidates.
    const bool swept_search = self_contact || !settings.cache_collision_queries || !reuse ||
                              cache.first_query || validate_trajectory;
    cache.first_query = false;
    if (!reuse) {
        tree.refit(second, b0, b1);
        cache.padding = settings.cache_collision_queries ? settings.collision_cache_padding : 0.0;
        cache.margin = margin;
        cache.a0 = a0; cache.a1 = a1; cache.b0 = b0; cache.b1 = b1;
        cache.candidates.clear();
        cache.candidates.resize(first.size());
        double query_margin = margin + 2.0 * cache.padding;
        if (settings.cache_collision_queries) {
            double scale = 1.0;
            for (const auto* positions : {&a0, &a1, &b0, &b1})
                for (const auto& point : *positions)
                    scale = std::max(scale, point.cwiseAbs().maxCoeff());
            query_margin += 64.0 * std::numeric_limits<double>::epsilon() * scale;
        }
        for (std::size_t i = 0; i < first.size(); ++i) {
            const auto& a = first[i];
            for (const auto index : tree.candidates(sweptBox(a, a0, a1),
                                                   query_margin)) {
                const auto& b = second[index];
                if (self_contact) {
                    // Incident features coincide by construction. Reject them
                    // before narrow phase, and keep one ordering per edge pair.
                    bool incident = false;
                    for (std::size_t j = 0; j < a.size(); ++j)
                        for (std::size_t k = 0; k < b.size(); ++k)
                            incident = incident || a.indices[j] == b.indices[k];
                    if (incident || (edge_edge && !(a.indices < b.indices))) continue;
                }
                cache.candidates[i].push_back({index, Vec3::Zero()});
            }
        }
    }
    // Refitting the current primitive bounds is cheap and does not require BVH
    // traversal. A padded candidate becomes active only after this exact test.
    std::vector<Box> second_boxes;
    second_boxes.reserve(second.size());
    for (const auto& feature : second) second_boxes.push_back(sweptBox(feature, b0, b1));
    const Eigen::Array3d error = Eigen::Array3d::Constant(-1.0);
    for (std::size_t i = 0; i < first.size(); ++i) {
        const auto& a = first[i];
        const auto first_box = sweptBox(a, a0, a1);
        for (auto& candidate : cache.candidates[i]) {
            if (!first_box.overlaps(second_boxes[candidate.index], margin)) continue;
            const auto& b = second[candidate.index];

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
            if (settings.cache_collision_queries && !self_contact && !validate_trajectory) {
                // Resolve endpoint proximity directly; refining a TOI for a
                // resting contact is unnecessary for XPBD position projection.
                // Preserve the admissible side using the SAME endpoint weights
                // evaluated at the start, so a crossed pair is pushed back.
                auto endpoint = makeContact(a, b, end, start, edge_edge, reverse, 1.0);
                const double endpoint_gap = reverse
                    ? evaluateContact(endpoint, b1, a1, settings.separation).gap
                    : evaluateContact(endpoint, a1, b1, settings.separation).gap;
                if (endpoint_gap <= settings.ccd_tolerance) {
                    const double initial_side = reverse
                        ? evaluateContact(endpoint, b0, a0, 0.0).gap
                        : evaluateContact(endpoint, a0, b0, 0.0).gap;
                    if (initial_side < 0.0) endpoint.normal = -endpoint.normal;
                    endpoint.self_contact = self_contact;
                    endpoint.time_of_impact.reset();
                    contacts.push_back(std::move(endpoint));
                    continue;
                }
                // Inside a certified candidate envelope, distant endpoints need
                // no repeated swept discovery on every solver iteration.
                // A newly created swept-only contact can be delayed. The final
                // mandatory full trajectory query detects any unresolved crossing
                // and triggers a smaller-substep retry, never silent acceptance.
                if (!swept_search) continue;
            }

            // Only a valid geometric certificate can suppress a scheduled CCD
            // query. Changing endpoints invalidates its proof automatically.
            if (settings.cache_collision_queries && !self_contact &&
                separatedOverStep(candidate.separating_direction, start, end, edge_edge, settings))
                continue;

            // Existing proximity is a contact at time zero, so it needs no root search.
            std::array<Vec3, 4> prior;
            for (int i = 0; i < 4; ++i) prior[i] = start[i] - 1e-4 * (end[i] - start[i]);
            auto initial = makeContact(a, b, start, prior, edge_edge, reverse, 0.0);
            const double initial_gap = reverse
                ? evaluateContact(initial, b0, a0, settings.separation).gap
                : evaluateContact(initial, a0, b0, settings.separation).gap;
            if (initial_gap <= settings.ccd_tolerance) {
                initial.self_contact = self_contact;
                contacts.push_back(std::move(initial));
                continue;
            }
            if (settings.cache_collision_queries && !self_contact) {
                const Vec3 direction = reverse ? -initial.normal : initial.normal;
                if (separatedOverStep(direction, start, end, edge_edge, settings)) {
                    candidate.separating_direction = direction;
                    continue;
                }
            }
            if (settings.cache_collision_queries && !self_contact &&
                certifySeparatedIntervals(start, end, edge_edge, settings))
                continue;
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
            auto contact = makeContact(a, b, impact, before, edge_edge, reverse, toi);
            contact.self_contact = self_contact;
            contacts.push_back(std::move(contact));
        }
    }
}

} // namespace

// Features are in the same sorted order as the uncached implementation. Although
// refitting may visit BVH leaves in a different order, candidates() sorts primitive
// IDs before narrow-phase tests. Full-query ordering stays deterministic. Scheduled
// obstacle discovery deliberately uses endpoint geometry; replay must use the
// contact records produced by that forward model, including absent impact times.
struct ContactDetector::Impl
{
    Features cloth_features;
    Features obstacle_features;
    std::vector<KinematicCollider::Triangle> obstacle_triangles;
    BoxTree cloth_faces;
    BoxTree cloth_edges;
    BoxTree obstacle_faces;
    BoxTree obstacle_edges;
    // Separate caches are required even when two queries share a target BVH:
    // vertex-face, reverse vertex-face, edge-edge, self VF and self EE have
    // different source features and motion envelopes.
    PairCache mesh_vf, mesh_fv, mesh_ee, self_vf, self_ee;

    explicit Impl(const mesh& cloth)
    {
        std::vector<KinematicCollider::Triangle> triangles;
        triangles.reserve(cloth.triangles.size());
        for (const auto& triangle : cloth.triangles) triangles.push_back(triangle.indices);
        cloth_features = makeFeatures(triangles);
    }

    void updateObstacle(const ColliderInput& collider)
    {
        // A caller may reuse this detector with a different obstacle. Connectivity
        // comparison also handles topology changes with an unchanged face count.
        if (obstacle_triangles != collider.triangles) {
            obstacle_triangles = collider.triangles;
            obstacle_features = makeFeatures(obstacle_triangles);
            obstacle_faces.clear();
            obstacle_edges.clear();
            mesh_vf = {};
            mesh_fv = {};
            mesh_ee = {};
        }
    }
};

ContactDetector::ContactDetector(const mesh& cloth) : impl_(std::make_unique<Impl>(cloth)) {}
ContactDetector::~ContactDetector() = default;

void ContactDetector::beginSubstep()
{
    impl_->mesh_vf.first_query = true;
    impl_->mesh_fv.first_query = true;
    impl_->mesh_ee.first_query = true;
    impl_->self_vf.first_query = true;
    impl_->self_ee.first_query = true;
}


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

std::vector<Contact> ContactDetector::detectMeshContacts(const Positions& start_positions,
    const Positions& end_positions, const ColliderInput& collider, const ContactSettings& settings,
    const std::vector<Contact>* known_contacts)
{
    std::set<PairKey> known;
    if (known_contacts) {
        for (const auto& contact : *known_contacts)
            if (!contact.self_contact) known.emplace(static_cast<int>(contact.cloth.feature_type), contact.cloth.indices,
                          static_cast<int>(contact.obstacle.feature_type), contact.obstacle.indices);
    }
    impl_->updateObstacle(collider);
    const auto& cloth_features = impl_->cloth_features;
    const auto& obstacle_features = impl_->obstacle_features;
    const auto& step = collider.step;
    std::vector<Contact> contacts;
    detectPairs(cloth_features.vertices, start_positions, end_positions,
                obstacle_features.faces, step.start_positions, step.end_positions,
                false, false, settings, known, contacts, impl_->obstacle_faces, impl_->mesh_vf, known_contacts == nullptr);
    detectPairs(obstacle_features.vertices, step.start_positions, step.end_positions,
                cloth_features.faces, start_positions, end_positions,
                false, true, settings, known, contacts, impl_->cloth_faces, impl_->mesh_fv, known_contacts == nullptr);
    detectPairs(cloth_features.edges, start_positions, end_positions,
                obstacle_features.edges, step.start_positions, step.end_positions,
                true, false, settings, known, contacts, impl_->obstacle_edges, impl_->mesh_ee, known_contacts == nullptr);
    return contacts;
}

std::vector<Contact> ContactDetector::detectSelfContacts(const Positions& start_positions,
    const Positions& end_positions, const ContactSettings& settings,
    const std::vector<Contact>* known_contacts)
{
    std::set<PairKey> known;
    if (known_contacts) {
        for (const auto& contact : *known_contacts)
            if (contact.self_contact)
                known.emplace(static_cast<int>(contact.cloth.feature_type), contact.cloth.indices,
                              static_cast<int>(contact.obstacle.feature_type), contact.obstacle.indices);
    }
    const auto& features = impl_->cloth_features;
    std::vector<Contact> contacts;
    // One vertex-face traversal covers all cloth vertices against all faces.
    // Both sides move, so CCD sees relative motion of both cloth features.
    detectPairs(features.vertices, start_positions, end_positions,
                features.faces, start_positions, end_positions,
                false, false, settings, known, contacts, impl_->cloth_faces, impl_->self_vf, known_contacts == nullptr, true);
    detectPairs(features.edges, start_positions, end_positions,
                features.edges, start_positions, end_positions,
                true, false, settings, known, contacts, impl_->cloth_edges, impl_->self_ee, known_contacts == nullptr, true);
    return contacts;
}

// Preserve standalone entry points. A temporary detector keeps one-shot callers
// safe without a persistent cache or any hidden global/thread-local state.
std::vector<Contact> detectMeshContacts(const mesh& cloth, const Positions& start_positions,
    const Positions& end_positions, const ColliderInput& collider, const ContactSettings& settings,
    const std::vector<Contact>* known_contacts)
{
    ContactDetector detector(cloth);
    return detector.detectMeshContacts(start_positions, end_positions, collider, settings, known_contacts);
}

std::vector<Contact> detectSelfContacts(const mesh& cloth, const Positions& start_positions,
    const Positions& end_positions, const ContactSettings& settings,
    const std::vector<Contact>* known_contacts)
{
    ContactDetector detector(cloth);
    return detector.detectSelfContacts(start_positions, end_positions, settings, known_contacts);
}

} // namespace diffxpbd
