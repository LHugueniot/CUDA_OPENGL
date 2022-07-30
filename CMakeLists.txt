# Common ======================================================================
cmake_minimum_required(VERSION 3.23)
#set(CUDA_HOST_COMPILER clang++)
#set(CMAKE_CUDA_COMPILER "C:/Program Files/LLVM/bin/clang++.exe")
set(CMAKE_CXX_STANDARD 17)

set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
add_compile_definitions(_USE_MATH_DEFINES)

set(PROJECTNAME "CUDA_OPENGL")
project(${PROJECTNAME} LANGUAGES CUDA CXX)

# Main App ====================================================================

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
    ${CUDA_OPENGL_SOURCE_DIR}/Camera.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/CuGlBuffer.cu
    ${CUDA_OPENGL_SOURCE_DIR}/Geometry.cu
    ${CUDA_OPENGL_SOURCE_DIR}/GLFWState.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/GLShader.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/main.cu
    ${CUDA_OPENGL_SOURCE_DIR}/MonoColourGLShader.cpp
    ${CUDA_OPENGL_SOURCE_DIR}/PlaneGLData.cpp
)
set(CUDA_OPENGL_ALL_FILES ${CUDA_OPENGL_HEADERS} ${CUDA_OPENGL_SOURCES})
add_executable(cuda_opengl_app ${CUDA_OPENGL_ALL_FILES})
target_include_directories(cuda_opengl_app PUBLIC ${CUDA_OPENGL_HEADER_DIR})
set_target_properties(cuda_opengl_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set up external includes and linking

set(EXTERNAL_LIB_DIRS
    "C:/Users/lucciano/Workspace/Libs/bin/glew-2.1.0-win32/glew-2.1.0/lib/Release/x64"
    "C:/Users/lucciano/Workspace/Libs/Bin/glfw-3.3.7.bin.WIN64/glfw-3.3.7.bin.WIN64"
)
list(APPEND CMAKE_PREFIX_PATH ${EXTERNAL_LIB_DIRS})

set(EXTERNAL_LIB_INCLUDES
    "C:/Users/lucciano/Workspace/Libs/bin/glew-2.1.0-win32/glew-2.1.0/include"
    "C:/Users/lucciano/Workspace/Libs/bin/eigen"
    "C:/Users/lucciano/Workspace/Libs/Bin/glm/GLM-0.9.9.8"
)

# OpenGL setup
find_package(OpenGL REQUIRED)
target_link_libraries(cuda_opengl_app ${OPENGL_LIBRARY})
message( "OpenGL included at :${OPENGL_LIBRARIES}")
message( "OpenGL lib can be found at :${OPENGL_gl_LIBRARY}")


# GLEW setup
find_library(GLEW_LIB "glew32")
target_link_libraries(cuda_opengl_app "${GLEW_LIB}")

# GLFW setup
find_package(GLFW3 REQUIRED)
target_link_libraries(cuda_opengl_app ${GLFW3_LIBRARIES})
target_include_directories(cuda_opengl_app PUBLIC ${GLFW3_INCLUDE_DIRS})

# CUDA setup
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

target_include_directories(cuda_opengl_app PUBLIC ${EXTERNAL_LIB_INCLUDES})

# Tests =======================================================================

include(CTest)

set(TEST_SOURCES
    test/main.cu
)

set(TEST_ALL_FILES ${CUDA_OPENGL_HEADERS} ${TEST_SOURCES})

add_executable(cuda_opengl_tests ${TEST_ALL_FILES})

target_include_directories(cuda_opengl_tests PUBLIC ${CUDA_OPENGL_HEADER_DIR})

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

# Link test executable against gtest & gtest_main
target_link_libraries(cuda_opengl_tests gtest gtest_main)
add_test(cuda_opengl_gtests cuda_opengl_tests)