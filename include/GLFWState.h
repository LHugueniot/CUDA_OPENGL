#ifndef GLFW_STATE_H
#define GLFW_STATE_H

#include <set>
#include <string>
#include <memory>

#include "Utils/OpenGL.h"
#include "Utils/General.h"

static int kGlMajorVersion = 3;
static int kGlMinorVersion = 2;
static uint kDefaultWindowWidth = 1280;
static uint kDefaultWindowHeight = 720;

struct Window
{
    Window() = default;
    ~Window() = default;
    std::string m_windowTitle;
    uint m_windowWidth;
    uint m_windowHeight;
    float m_xScroll;
    float m_yScroll;
    GLFWwindow* m_glfwWindow;
    bool m_initSuccessful;
};
using WinPtr = std::shared_ptr<Window>;

struct GLFWState
{
    int m_glMajorVersion;
    int m_glMinorVersion;
    std::set<WinPtr> m_windows;
    bool m_initSuccessful;
};
using GLFWStatePtr = std::unique_ptr<GLFWState>;

GLFWState setupGLFW(
    int glMajorVersion=kGlMajorVersion,
    int glMinorVersion=kGlMinorVersion);

bool shouldQuit(GLFWState const& state);

void teardown(GLFWState& state);

bool addWindow(GLFWState& state, WinPtr window);

bool removeWindow(GLFWState& state, WinPtr window);

WinPtr createWindow(std::string const& windowTitle,
    uint windowWidth=kDefaultWindowWidth,
    uint windowHeight=kDefaultWindowHeight);

#endif /* GLFW_STATE_H */