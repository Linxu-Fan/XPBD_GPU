#pragma once

#include "diffxpbd/xpbd.hpp"

#include <memory>

namespace diffxpbd {

class Viewer
{
public:
    explicit Viewer(const mesh& rest_mesh);
    ~Viewer();
    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;

    bool pollEvents();
    bool shouldAdvance();
    void draw(const mesh& m, const state& s, int step, double time, bool finished,
              const ColliderInput* collider = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace diffxpbd
