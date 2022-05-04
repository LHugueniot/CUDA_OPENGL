cmake_minimum_required(VERSION 3.23)
#set(CUDA_HOST_COMPILER clang++)
#set(CMAKE_CUDA_COMPILER "C:/Program Files/LLVM/bin/clang++.exe")
set(CMAKE_CXX_STANDARD 17)

set(PROJECTNAME "CUDA_OPENGL")
project(${PROJECTNAME} LANGUAGES CUDA CXX)

include(CTest)

# Set up internal includes and linking
set(HEADER_DIR "include")
set(SOURCE_DIR "src")
set(HEADERS
    ${HEADER_DIR}/Utils/Cuda.cuh
    ${HEADER_DIR}/Utils/CudaGL.cuh
    ${HEADER_DIR}/Utils/Eigen.h
    ${HEADER_DIR}/Utils/General.h
    ${HEADER_DIR}/Utils/Math.h
    ${HEADER_DIR}/Utils/OpenGL.h
    ${HEADER_DIR}/Camera.h
    ${HEADER_DIR}/CuGlBuffer.cuh
    ${HEADER_DIR}/Geometry.cuh
    ${HEADER_DIR}/GLFWState.h
    ${HEADER_DIR}/GLShader.h
    ${HEADER_DIR}/MacGrid.cuh
    ${HEADER_DIR}/MonoColourGLShader.h
    ${HEADER_DIR}/PlaneGLData.h
)
set(SOURCES
    ${SOURCE_DIR}/Camera.cpp
    ${SOURCE_DIR}/CuGlBuffer.cu
    ${SOURCE_DIR}/Geometry.cu
    ${SOURCE_DIR}/GLFWState.cpp
    ${SOURCE_DIR}/GLShader.cpp
    ${SOURCE_DIR}/main.cu
    ${SOURCE_DIR}/MonoColourGLShader.cpp
    ${SOURCE_DIR}/PlaneGLData.cpp
)
set(ALL_FILES ${HEADERS} ${SOURCES})
add_executable(cuda_opengl_app ${ALL_FILES})
target_include_directories(cuda_opengl_app PUBLIC ${HEADER_DIR})
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