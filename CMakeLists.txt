# Common ======================================================================
cmake_minimum_required(VERSION 3.25)
include(CMakePrintHelpers)

# Utility functions ===========================================================
function(extract_macro_val MACRO_NAME HEADER_FILE)
    string(REGEX MATCH "#define ${MACRO_NAME} [0-9]+" _macro_def "${HEADER_FILE}")
    string(REGEX MATCH "[0-9]+" _macro_val "${_macro_def}")

    set(extract_macro_val_RETURN ${_macro_val} PARENT_SCOPE)
endfunction()

# set(CUDA_HOST_COMPILER clang++)
# set(CMAKE_CUDA_COMPILER "C:/Program Files/LLVM/bin/clang++.exe")
set(CMAKE_CXX_STANDARD 17)

add_compile_definitions(_USE_MATH_DEFINES)

set(PROJECTNAME "CUDA_OPENGL")
project(${PROJECTNAME} LANGUAGES CUDA CXX)

# Main App Lib ================================================================

# Set up internal includes and linking
set(CUDA_OPENGL_HEADER_DIR "include")
set(CUDA_OPENGL_SOURCE_DIR "src")
set(CUDA_OPENGL_HEADERS
    ${CUDA_OPENGL_HEADER_DIR}/PositionBasedDynamics/PBDSolver.cuh
    ${CUDA_OPENGL_HEADER_DIR}/Utils/Cuda.cuh
    ${CUDA_OPENGL_HEADER_DIR}/Utils/CudaGL.cuh
    ${CUDA_OPENGL_HEADER_DIR}/Utils/Eigen.h
    ${CUDA_OPENGL_HEADER_DIR}/Utils/General.h
    ${CUDA_OPENGL_HEADER_DIR}/Utils/Math.h
    ${CUDA_OPENGL_HEADER_DIR}/Utils/OpenGL.h
    ${CUDA_OPENGL_HEADER_DIR}/Utils/Mesh.h
    ${CUDA_OPENGL_HEADER_DIR}/Camera.h
    ${CUDA_OPENGL_HEADER_DIR}/CuGlBuffer.cuh
    ${CUDA_OPENGL_HEADER_DIR}/CuGlGeometry.cuh
    ${CUDA_OPENGL_HEADER_DIR}/GeometryViewer.cuh
    ${CUDA_OPENGL_HEADER_DIR}/Geometry.cuh
    ${CUDA_OPENGL_HEADER_DIR}/GLFWState.h
    ${CUDA_OPENGL_HEADER_DIR}/LoadGeometry.cuh
    ${CUDA_OPENGL_HEADER_DIR}/GLShader.h
    ${CUDA_OPENGL_HEADER_DIR}/MacGrid.cuh
    ${CUDA_OPENGL_HEADER_DIR}/MonoColourGLShader.h
    ${CUDA_OPENGL_HEADER_DIR}/PlaneGLData.h
)
set(CUDA_OPENGL_SOURCES
    ${CUDA_OPENGL_SOURCE_DIR}/PositionBasedDynamics/PBDSolver.cu
    ${CUDA_OPENGL_SOURCE_DIR}/Utils/Mesh.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/Utils/Eigen.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/Utils/OpenGL.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/Camera.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/CuGlBuffer.cu
    ${CUDA_OPENGL_SOURCE_DIR}/CuGlGeometry.cu
    ${CUDA_OPENGL_SOURCE_DIR}/GeometryViewer.cu
    ${CUDA_OPENGL_SOURCE_DIR}/Geometry.cu
    ${CUDA_OPENGL_SOURCE_DIR}/GLFWState.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/GLShader.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/LoadGeometry.cu
    ${CUDA_OPENGL_SOURCE_DIR}/MonoColourGLShader.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/PlaneGLData.cpp
)

# set(CUDA_OPENGL_ALL_FILES ${CUDA_OPENGL_HEADERS} ${CUDA_OPENGL_SOURCES})
add_library(cuda_opengl_lib ${CUDA_OPENGL_SOURCES})

target_compile_options(cuda_opengl_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    >)
target_include_directories(cuda_opengl_lib PUBLIC ${CUDA_OPENGL_HEADER_DIR})
set_target_properties(cuda_opengl_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Assimp setup
find_package(assimp CONFIG REQUIRED)

# Eigen3 setup
find_package(Eigen3 CONFIG REQUIRED)

# ImGui setup
find_package(imgui CONFIG REQUIRED)

# OpenGL setup
find_package(OpenGL REQUIRED)

# glew setup
find_package(glew REQUIRED)

message(STATUS ${GLEW})

# glfw3 setup
find_package(glfw3 CONFIG REQUIRED)

# CUDA setup
set(CUDA_REQUIRED_VERSION "11.6.0")
set(CUDA_PATHS
    ENV CUDA_PATH
    ENV CUDA_HOME
    ENV CUDA_BIN_PATH
    ENV CUDA_PATH_V11_6
)
find_package(CUDA REQUIRED)

target_link_libraries(cuda_opengl_lib
    PRIVATE
    OpenGL::GL
    glfw
    Eigen3::Eigen
    assimp::assimp

    PUBLIC
    imgui::imgui
    GLEW::glew
)

get_target_property(GLEW_INCLUDE_DIR GLEW::glew INTERFACE_INCLUDE_DIRECTORIES)

cmake_print_properties(TARGETS imgui::imgui PROPERTIES
    LOCATION INTERFACE_INCLUDE_DIRECTORIES)

message(STATUS ${GLEW_INCLUDE_DIR})

target_include_directories(cuda_opengl_lib PRIVATE ${GLEW_INCLUDE_DIR})

# Main App ====================================================================
add_executable(cuda_opengl_app)
set_target_properties(cuda_opengl_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_sources(cuda_opengl_app PUBLIC ${CUDA_OPENGL_SOURCE_DIR}/main.cu)
target_link_libraries(cuda_opengl_app cuda_opengl_lib)

target_compile_options(cuda_opengl_app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    >)

# Tests =======================================================================
include(CTest)
enable_testing()

set(TEST_SOURCE_DIR "test")

# Create test executable
set(TEST_SOURCES
)
set(TEST_ALL_FILES ${TEST_SOURCES})
add_executable(cuda_opengl_tests_main ${TEST_SOURCE_DIR}/main.cu ${TEST_SOURCES})

target_compile_options(cuda_opengl_tests_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    >)

set_target_properties(cuda_opengl_tests_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link application library
target_link_libraries(cuda_opengl_tests_main PRIVATE cuda_opengl_lib)

# GLM setup
find_package(glm CONFIG REQUIRED)

# Gtest setup
find_package(GTest CONFIG REQUIRED)

target_link_libraries(
    cuda_opengl_tests_main
    PRIVATE
    GTest::gtest
    GTest::gtest_main
    glm::glm
)
add_test(
    NAME cuda_opengl_tests
    COMMAND cuda_opengl_tests_main.exe)

# include(GoogleTest)
# gtest_discover_tests(cuda_opengl_tests_main)
