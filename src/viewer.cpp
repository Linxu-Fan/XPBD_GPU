#include "diffxpbd/viewer.hpp"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <windowsx.h>
#include <GL/gl.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace diffxpbd {

struct Viewer::Impl
{
    enum class DragMode { None, Orbit, Pan };
    HWND window = nullptr;
    HDC device = nullptr;
    HGLRC context = nullptr;
    bool closing = false;
    bool paused = false;
    bool single_step = false;
    bool wireframe = true;
    double yaw = -25.0;
    double pitch = 15.0;
    double distance = 1.0;
    double radius = 1.0;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    Eigen::Vector3d rest_center = Eigen::Vector3d::Zero();
    DragMode drag_mode = DragMode::None;
    int mouse_x = 0;
    int mouse_y = 0;

    static LRESULT CALLBACK windowProc(HWND window, UINT message, WPARAM key, LPARAM data)
    {
        auto* self = reinterpret_cast<Impl*>(GetWindowLongPtrW(window, GWLP_USERDATA));
        if (message == WM_NCCREATE) {
            const auto* create = reinterpret_cast<CREATESTRUCTW*>(data);
            self = static_cast<Impl*>(create->lpCreateParams);
            SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
        }
        if (self) {
            if (message == WM_LBUTTONDOWN || message == WM_MBUTTONDOWN || message == WM_RBUTTONDOWN) {
                self->drag_mode = message == WM_LBUTTONDOWN ? DragMode::Orbit : DragMode::Pan;
                self->mouse_x = GET_X_LPARAM(data);
                self->mouse_y = GET_Y_LPARAM(data);
                SetCapture(window);
                SetFocus(window);
                return 0;
            }
            if (message == WM_LBUTTONUP || message == WM_MBUTTONUP || message == WM_RBUTTONUP) {
                self->drag_mode = DragMode::None;
                if (GetCapture() == window) {
                    ReleaseCapture();
                }
                return 0;
            }
            if (message == WM_CAPTURECHANGED) {
                self->drag_mode = DragMode::None;
                return 0;
            }
            if (message == WM_MOUSEMOVE && self->drag_mode != DragMode::None) {
                const int x = GET_X_LPARAM(data);
                const int y = GET_Y_LPARAM(data);
                const int dx = x - self->mouse_x;
                const int dy = y - self->mouse_y;
                self->mouse_x = x;
                self->mouse_y = y;
                if (self->drag_mode == DragMode::Orbit) {
                    self->yaw += dx * 0.4;
                    self->pitch = std::clamp(self->pitch + dy * 0.4, -85.0, 85.0);
                } else {
                    RECT bounds;
                    GetClientRect(window, &bounds);
                    const double radians = 3.141592653589793 / 180.0;
                    const double scale = 2.0 * self->distance * std::tan(22.5 * radians) /
                        std::max(1L, bounds.bottom);
                    const Eigen::Matrix3d rotation =
                        (Eigen::AngleAxisd(self->pitch * radians, Eigen::Vector3d::UnitX()) *
                         Eigen::AngleAxisd(self->yaw * radians, Eigen::Vector3d::UnitY())).toRotationMatrix();
                    self->center -= rotation.transpose() * Eigen::Vector3d(dx * scale, -dy * scale, 0.0);
                }
                return 0;
            }
            if (message == WM_MOUSEWHEEL) {
                const double ticks = static_cast<double>(GET_WHEEL_DELTA_WPARAM(key)) / WHEEL_DELTA;
                self->distance = std::clamp(self->distance * std::pow(0.85, ticks),
                                           self->radius * 0.2, self->radius * 100.0);
                return 0;
            }
            if (message == WM_CLOSE) {
                self->closing = true;
                return 0;
            }
            if (message == WM_KEYDOWN) {
                if (data & (1LL << 30)) {
                    return 0;
                }
                switch (key) {
                case VK_ESCAPE: self->closing = true; break;
                case VK_SPACE: self->paused = !self->paused; self->single_step = false; break;
                case 'N': self->paused = true; self->single_step = true; break;
                case 'W': self->wireframe = !self->wireframe; break;
                case 'R':
                    self->yaw = -25.0;
                    self->pitch = 15.0;
                    self->distance = self->radius * 3.5;
                    self->center = self->rest_center;
                    break;
                }
                return 0;
            }
        }
        return DefWindowProcW(window, message, key, data);
    }

    ~Impl()
    {
        if (context) {
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(context);
        }
        if (device) {
            ReleaseDC(window, device);
        }
        if (window) {
            DestroyWindow(window);
        }
    }
};

Viewer::Viewer(const mesh& rest_mesh) : impl_(std::make_unique<Impl>())
{
    if (rest_mesh.vertices.empty()) {
        throw std::runtime_error("Viewer requires a nonempty mesh");
    }
    Eigen::Vector3d minimum = rest_mesh.vertices.front();
    Eigen::Vector3d maximum = minimum;
    for (const auto& position : rest_mesh.vertices) {
        minimum = minimum.cwiseMin(position);
        maximum = maximum.cwiseMax(position);
    }
    impl_->center = (minimum + maximum) * 0.5;
    impl_->rest_center = impl_->center;
    impl_->radius = std::max((maximum - minimum).norm() * 0.5, 0.01);
    impl_->distance = impl_->radius * 3.5;

    const auto instance = GetModuleHandleW(nullptr);
    WNDCLASSW window_class{};
    window_class.style = CS_OWNDC;
    window_class.lpfnWndProc = Impl::windowProc;
    window_class.hInstance = instance;
    window_class.hCursor = LoadCursor(nullptr, IDC_ARROW);
    window_class.lpszClassName = L"DiffXPBDOpenGL";
    if (!RegisterClassW(&window_class) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        throw std::runtime_error("Failed to register the viewer window");
    }
    impl_->window = CreateWindowW(window_class.lpszClassName, L"DiffXPBD",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1000, 800,
        nullptr, nullptr, instance, impl_.get());
    if (!impl_->window) {
        throw std::runtime_error("Failed to create the viewer window");
    }
    impl_->device = GetDC(impl_->window);
    PIXELFORMATDESCRIPTOR format{};
    format.nSize = sizeof(format);
    format.nVersion = 1;
    format.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    format.iPixelType = PFD_TYPE_RGBA;
    format.cColorBits = 24;
    format.cDepthBits = 24;
    const int pixel_format = ChoosePixelFormat(impl_->device, &format);
    if (!pixel_format || !SetPixelFormat(impl_->device, pixel_format, &format)) {
        throw std::runtime_error("Failed to configure an OpenGL pixel format");
    }
    impl_->context = wglCreateContext(impl_->device);
    if (!impl_->context || !wglMakeCurrent(impl_->device, impl_->context)) {
        throw std::runtime_error("Failed to create an OpenGL context");
    }
    ShowWindow(impl_->window, SW_SHOW);
}

Viewer::~Viewer() = default;

bool Viewer::pollEvents()
{
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        if (message.message == WM_QUIT) {
            impl_->closing = true;
        }
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    return !impl_->closing;
}

bool Viewer::shouldAdvance()
{
    const bool advance = !impl_->paused || impl_->single_step;
    impl_->single_step = false;
    return advance;
}

void Viewer::draw(const mesh& m, const state& s, int step, double time, bool finished,
                  const ColliderInput* collider)
{
    RECT bounds;
    GetClientRect(impl_->window, &bounds);
    const int width = bounds.right;
    const int height = bounds.bottom;
    if (width <= 0 || height <= 0) {
        return;
    }
    glViewport(0, 0, width, height);
    glClearColor(0.91f, 0.93f, 0.96f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    const double near_plane = impl_->radius * 0.01;
    const double top = near_plane * std::tan(22.5 * 3.141592653589793 / 180.0);
    const double right = top * static_cast<double>(width) / height;
    glFrustum(-right, right, -top, top, near_plane, impl_->radius * 200.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslated(0.0, 0.0, -impl_->distance);
    glRotated(impl_->pitch, 1.0, 0.0, 0.0);
    glRotated(impl_->yaw, 0.0, 1.0, 0.0);
    glTranslated(-impl_->center.x(), -impl_->center.y(), -impl_->center.z());

    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);
    glBegin(GL_TRIANGLES);
    const Eigen::Vector3d light = Eigen::Vector3d(0.4, 0.8, 0.6).normalized();
    for (const auto& triangle : m.triangles) {
        const auto [a, b, c] = triangle.indices;
        const Eigen::Vector3d normal =
            (s.position[b] - s.position[a]).cross(s.position[c] - s.position[a]).normalized();
        const double shade = 0.35 + 0.65 * std::abs(normal.dot(light));
        glColor3d(0.2 * shade, 0.6 * shade, 0.9 * shade);
        for (const auto index : triangle.indices) {
            glVertex3dv(s.position[index].data());
        }
    }
    glEnd();
    glDisable(GL_POLYGON_OFFSET_FILL);
    if (impl_->wireframe) {
        glColor3d(0.12, 0.2, 0.28);
        glBegin(GL_LINES);
        for (const auto& triangle : m.triangles) {
            for (int edge = 0; edge < 3; ++edge) {
                glVertex3dv(s.position[triangle.indices[edge]].data());
                glVertex3dv(s.position[triangle.indices[(edge + 1) % 3]].data());
            }
        }
        glEnd();
    }
    if (collider) {
        glBegin(GL_TRIANGLES);
        const auto& positions = collider->step.end_positions;
        for (const auto& [a, b, c] : collider->triangles) {
            const Eigen::Vector3d normal =
                (positions[b] - positions[a]).cross(positions[c] - positions[a]).normalized();
            const double shade = 0.4 + 0.6 * std::abs(normal.dot(light));
            glColor3d(0.65 * shade, 0.65 * shade, 0.65 * shade);
            glVertex3dv(positions[a].data());
            glVertex3dv(positions[b].data());
            glVertex3dv(positions[c].data());
        }
        glEnd();
    }
    glPointSize(8.0f);
    glColor3d(0.9, 0.15, 0.1);
    glBegin(GL_POINTS);
    for (std::size_t i = 0; i < s.pinned_positions.size(); ++i) {
        if (s.pinned_positions[i]) {
            glVertex3dv(s.position[i].data());
        }
    }
    glEnd();
    SwapBuffers(impl_->device);

    std::wostringstream title;
    title << L"DiffXPBD | Step " << step << L" | " << std::fixed << std::setprecision(3)
          << time << L" s | " << (finished ? L"Complete" : impl_->paused ? L"Paused" : L"Simulating")
          << L" | Left drag: orbit | Right/middle drag: pan | Wheel: zoom"
          << L" | Space: pause | N: step | W: edges | R: reset";
    SetWindowTextW(impl_->window, title.str().c_str());
}

} // namespace diffxpbd
