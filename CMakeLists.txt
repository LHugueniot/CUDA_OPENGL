# Common ======================================================================
cmake_minimum_required(VERSION 3.25)
include(CMakePrintHelpers)

# Utility functions ===========================================================
function(extract_macro_val MACRO_NAME HEADER_FILE)
    string(REGEX MATCH "#define ${MACRO_NAME} [0-9]+"
        _macro_def "${HEADER_FILE}")
    string(REGEX MATCH "[0-9]+" _macro_val "${_macro_def}")

    set(extract_macro_val_RETURN ${_macro_val} PARENT_SCOPE)
endfunction()

set(CMAKE_CXX_STANDARD 17)

add_compile_definitions(_USE_MATH_DEFINES)

set(PROJECTNAME "CUDA_OPENGL")
project(${PROJECTNAME} LANGUAGES CUDA CXX)

# Cuda Opengl module libs and dependencies ====================================

# Assimp setup
find_package(assimp CONFIG REQUIRED)

# Eigen3 setup
find_package(Eigen3 CONFIG REQUIRED)

# ImGui setup
find_package(imgui CONFIG REQUIRED)

# OpenGL setup
find_package(OpenGL REQUIRED)

# glew setup
find_package(GLEW REQUIRED)

# glfw3 setup
find_package(glfw3 CONFIG REQUIRED)

add_subdirectory(UTILS)
add_subdirectory(VIEWER)
add_subdirectory(GEOMETRY)
add_subdirectory(SIMULATION)
add_subdirectory(UI)

# CUDA setup
set(CUDA_REQUIRED_VERSION "11.6.0")
set(CUDA_PATHS
    ENV CUDA_PATH
    ENV CUDA_HOME
    ENV CUDA_BIN_PATH
    ENV CUDA_PATH_V11_6
)
find_package(CUDA REQUIRED)

# get_target_property(GLEW_INCLUDE_DIR GLEW::glew
# INTERFACE_INCLUDE_DIRECTORIES)

# cmake_print_properties(TARGETS imgui::imgui PROPERTIES
# LOCATION INTERFACE_INCLUDE_DIRECTORIES)

# Main App ====================================================================

# Create executable
add_executable(cuda_opengl_app)

# Pre setup
set_target_properties(cuda_opengl_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(cuda_opengl_app PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    >)

# Sources
target_sources(cuda_opengl_app PUBLIC MAIN/src/main.cu)

# Lib linking
target_link_libraries(
    cuda_opengl_app

    PRIVATE
    Utils
    Viewer
    Geometry
    Sim
    Ui
)

# Asset setup
set(ASSETS_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ASSETS)
file(
    COPY ${CMAKE_CURRENT_SOURCE_DIR}/ASSETS/
    DESTINATION ${ASSETS_DIRECTORY}
)
target_compile_definitions(
    cuda_opengl_app

    PUBLIC
    ASSETS_DIRECTORY=${ASSETS_DIRECTORY}
)

# Tests =======================================================================
add_subdirectory(TESTS)
