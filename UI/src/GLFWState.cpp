#include <set>
#include <map>
#include <string>
#include <stdexcept>

#include "Utils/OpenGL.h"
#include "Utils/General.h"

#include "Ui/GLFWState.h"

static const std::map<int, std::string> kGLFWErrorCodeToString {
    {GLFW_NOT_INITIALIZED, "GLFW_NOT_INITIALIZED - 0x00010001"},
    {GLFW_NO_CURRENT_CONTEXT, "GLFW_NO_CURRENT_CONTEXT - 0x00010002"},
    {GLFW_INVALID_ENUM, "GLFW_INVALID_ENUM - 0x00010003"},
    {GLFW_INVALID_VALUE, "GLFW_INVALID_VALUE - 0x00010004"},
    {GLFW_OUT_OF_MEMORY, "GLFW_OUT_OF_MEMORY - 0x00010005"},
    {GLFW_API_UNAVAILABLE, "GLFW_API_UNAVAILABLE - 0x00010006"},
    {GLFW_VERSION_UNAVAILABLE, "GLFW_VERSION_UNAVAILABLE - 0x00010007"},
    {GLFW_PLATFORM_ERROR, "GLFW_PLATFORM_ERROR - 0x00010008"},
    {GLFW_FORMAT_UNAVAILABLE, "GLFW_FORMAT_UNAVAILABLE - 0x00010009"}
};

void _handleError(int error, const char* description);

void _windowClosing(GLFWwindow* window);

void _keyPressed(GLFWwindow* window,
    int key, int scancode, int action, int mods);

void _framebufferSizeChanged(GLFWwindow* window, int width, int height);

void _scrolled(GLFWwindow* window, double xoffset, double yoffset);


GLFWState setupGLFW(int glMajorVersion, int glMinorVersion)
{
    GLFWState state{glMajorVersion, glMinorVersion, std::set<WinPtr>(), false};

    glfwSetErrorCallback(_handleError);

    if (!glfwInit())
    {
        // Initialization failed
        fprintf(stderr, "GLFW initialization failed.\n");
        state.m_initSuccessful = false;
    }
    else
    {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, glMajorVersion);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, glMinorVersion);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        fprintf(stdout, "GLFW initialization successful.\n");

        state.m_initSuccessful = true;
    }

    return std::move(state);
}

bool shouldQuit(GLFWState const& state)
{
    bool quit = true;
    for (auto window : state.m_windows)
    {
        int winShouldClose = glfwWindowShouldClose(window->m_glfwWindow);
        //fprintf(stdout, "Should window %s close: %i.\n",
        //    window->m_windowTitle.c_str(), winShouldClose);
        quit = quit && winShouldClose;
    }
    return quit;
}

void teardown(GLFWState& state)
{
    if (!state.m_initSuccessful){
        fprintf(stderr, "Teardown unsuccessful, GLFW initialization failed.\n");
        return;
    }

    for (auto window : state.m_windows)
    {
        glfwDestroyWindow(window->m_glfwWindow);
    }

    state.m_windows.clear();

    glfwTerminate();
}

static std::map<GLFWwindow*, WinPtr> g_windowToContext;

bool addWindow(GLFWState& state, WinPtr window)
{
    auto windowIt = state.m_windows.find(window);
    if (windowIt != state.m_windows.end())
        return false;
    state.m_windows.insert(window);
    g_windowToContext[window->m_glfwWindow] = window;
    return true;
}

bool removeWindow(GLFWState& state, WinPtr window)
{
    auto windowIt = state.m_windows.find(window);
    if (windowIt == state.m_windows.end())
        return false;
    state.m_windows.erase(window);
    g_windowToContext.erase(window->m_glfwWindow);
    return true;
}

WinPtr createWindow(std::string const& windowTitle,
    uint windowWidth, uint windowHeight)
{
    WinPtr window = std::make_shared<Window>();
    window->m_windowTitle = windowTitle;
    window->m_windowWidth = windowWidth;
    window->m_windowHeight = windowHeight;
    window->m_xScroll = 0;
    window->m_yScroll = 0;
    window->m_glfwWindow = glfwCreateWindow(windowWidth,
        windowHeight, windowTitle.c_str(), NULL, NULL);
    window->m_initSuccessful = false;

    if (window->m_glfwWindow)
    {
        glfwMakeContextCurrent(window->m_glfwWindow);

        glfwSetFramebufferSizeCallback(window->m_glfwWindow, _framebufferSizeChanged);

        glfwSetWindowCloseCallback(window->m_glfwWindow, _windowClosing);

        glfwSetKeyCallback(window->m_glfwWindow, _keyPressed);

        glfwSetScrollCallback(window->m_glfwWindow, _scrolled);

        window->m_initSuccessful = true;
    }
    else
    {
        window->m_initSuccessful = false;
    }

    return std::move(window);
}

void _handleError(int error, const char* description)
{
    std::string errorCodeMessage = kGLFWErrorCodeToString.at(error);
    fprintf(stderr, "Error %s: %s\n", errorCodeMessage.c_str(), description);
    throw std::runtime_error("Error thrown.");
}

void _windowClosing(GLFWwindow* window)
{
    auto windowIt = g_windowToContext.find(window);
    if(windowIt == g_windowToContext.end())
        return;
    auto windowTitle = windowIt->second->m_windowTitle;
    fprintf(stdout, "Closing window: %s\n", windowTitle.c_str());
}

void _keyPressed(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto windowIt = g_windowToContext.find(window);
    if(windowIt == g_windowToContext.end())
        return;
    auto windowTitle = windowIt->second->m_windowTitle;
    //fprintf(stdout, "Key pressed for: %s\n", windowTitle.c_str());
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void _scrolled(GLFWwindow* window, double xoffset, double yoffset)
{
    fprintf(stdout, "Scrolling in: x(%f) y(%f)\n", xoffset, yoffset);
    auto windowIt = g_windowToContext.find(window);
    if(windowIt == g_windowToContext.end())
        return;
    windowIt->second->m_xScroll = xoffset;
    windowIt->second->m_yScroll = yoffset;
}

void _framebufferSizeChanged(GLFWwindow* window, int width, int height)
{
    auto windowIt = g_windowToContext.find(window);
    if(windowIt == g_windowToContext.end())
        return;
    auto windowTitle = windowIt->second->m_windowTitle;
    fprintf(stdout, "Window size changed for: %s\n", windowTitle.c_str());
    // make sure the viewport matches the new window dimensions
    glViewport(0, 0, width, height);
    windowIt->second->m_windowWidth = width;
    windowIt->second->m_windowHeight = height;
}
