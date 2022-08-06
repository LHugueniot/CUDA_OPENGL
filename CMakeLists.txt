# Common ======================================================================
cmake_minimum_required(VERSION 3.23)
#set(CUDA_HOST_COMPILER clang++)
#set(CMAKE_CUDA_COMPILER "C:/Program Files/LLVM/bin/clang++.exe")
set(CMAKE_CXX_STANDARD 17)

add_compile_definitions(_USE_MATH_DEFINES)

set(PROJECTNAME "CUDA_OPENGL")
project(${PROJECTNAME} LANGUAGES CUDA CXX)

# Main App Lib ================================================================

# Set up internal includes and linking
set(CUDA_OPENGL_HEADER_DIR "include")
set(CUDA_OPENGL_SOURCE_DIR "src")
set(CUDA_OPENGL_HEADERS
    ${CUDA_OPENGL_HEADER_DIR}/Utils/Cuda.cuh
    ${CUDA_OPENGL_HEADER_DIR}/Utils/CudaGL.cuh
    ${CUDA_OPENGL_HEADER_DIR}/Utils/Eigen.h
    ${CUDA_OPENGL_HEADER_DIR}/Utils/General.h
    ${CUDA_OPENGL_HEADER_DIR}/Utils/Math.h
    ${CUDA_OPENGL_HEADER_DIR}/Utils/OpenGL.h
    ${CUDA_OPENGL_HEADER_DIR}/Camera.h
    ${CUDA_OPENGL_HEADER_DIR}/CuGlBuffer.cuh
    ${CUDA_OPENGL_HEADER_DIR}/Geometry.cuh
    ${CUDA_OPENGL_HEADER_DIR}/GLFWState.h
    ${CUDA_OPENGL_HEADER_DIR}/GLShader.h
    ${CUDA_OPENGL_HEADER_DIR}/MacGrid.cuh
    ${CUDA_OPENGL_HEADER_DIR}/MonoColourGLShader.h
    ${CUDA_OPENGL_HEADER_DIR}/PlaneGLData.h
)
set(CUDA_OPENGL_SOURCES
    ${CUDA_OPENGL_SOURCE_DIR}/Utils/Eigen.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/Camera.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/CuGlBuffer.cu
    ${CUDA_OPENGL_SOURCE_DIR}/Geometry.cu
    ${CUDA_OPENGL_SOURCE_DIR}/GLFWState.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/GLShader.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/MonoColourGLShader.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/PlaneGLData.cpp
)
#set(CUDA_OPENGL_ALL_FILES ${CUDA_OPENGL_HEADERS} ${CUDA_OPENGL_SOURCES})

add_library(cuda_opengl_lib ${CUDA_OPENGL_SOURCES})

target_compile_options(cuda_opengl_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    >)

target_include_directories(cuda_opengl_lib PUBLIC ${CUDA_OPENGL_HEADER_DIR})
set_target_properties(cuda_opengl_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set up external includes and linking
set(EXTERNAL_LIB_DIRS
    C:/Users/lucciano/Workspace/Libs/bin/glew-2.1.0-win32/glew-2.1.0/lib/Release/x64
    C:/Users/lucciano/Workspace/Libs/Bin/glfw-3.3.7.bin.WIN64/glfw-3.3.7.bin.WIN64
)
list(APPEND CMAKE_PREFIX_PATH ${EXTERNAL_LIB_DIRS})

set(EXTERNAL_LIB_INCLUDES
    C:/Users/lucciano/Workspace/Libs/bin/glew-2.1.0-win32/glew-2.1.0/include
    C:/Users/lucciano/Workspace/Libs/bin/eigen
)

# OpenGL setup
find_package(OpenGL REQUIRED)
target_link_libraries(cuda_opengl_lib ${OPENGL_LIBRARY})

# GLEW setup
find_library(GLEW_LIB "glew32")
target_link_libraries(cuda_opengl_lib "${GLEW_LIB}")

# GLFW setup
find_package(GLFW3 REQUIRED)
target_link_libraries(cuda_opengl_lib ${GLFW3_LIBRARIES})
target_include_directories(cuda_opengl_lib PUBLIC ${GLFW3_INCLUDE_DIRS})

# CUDA setup
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

target_include_directories(cuda_opengl_lib PUBLIC ${EXTERNAL_LIB_INCLUDES})

# Main App ====================================================================

add_executable(cuda_opengl_app)
set_target_properties(cuda_opengl_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_sources(cuda_opengl_app PUBLIC ${CUDA_OPENGL_SOURCE_DIR}/main.cu)
target_link_libraries(cuda_opengl_app cuda_opengl_lib)

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

# Include test includes
set(EXTERNAL_TEST_INCLUDES C:/Users/lucciano/Workspace/Libs/Bin/glm/GLM-0.9.9.8)
target_include_directories(cuda_opengl_tests_main PUBLIC ${EXTERNAL_TEST_INCLUDES})

# Get and link gtest libs
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

target_link_libraries(
    cuda_opengl_tests_main
    PRIVATE
    GTest::gtest_main
)
add_test(
    NAME cuda_opengl_tests
    COMMAND cuda_opengl_tests_main.exe)

#include(GoogleTest)
#gtest_discover_tests(cuda_opengl_tests_main)
