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
    ${HEADER_DIR}/Camera.h
    ${HEADER_DIR}/CuGlBuffer.cuh
    ${HEADER_DIR}/Geometry.cuh
    ${HEADER_DIR}/GLShader.h
    ${HEADER_DIR}/MacGrid.cuh
    ${HEADER_DIR}/MonoColourGLShader.h
    ${HEADER_DIR}/PlaneGLData.h
    ${HEADER_DIR}/Utils.cuh
)

set(SOURCES
    ${SOURCE_DIR}/Camera.cpp
    ${SOURCE_DIR}/CuGlBuffer.cu
    ${SOURCE_DIR}/Geometry.cu
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
    "C:/Users/lucciano/Workspace/Libs/bin/SDL2-devel-2.0.20-VC/SDL2-2.0.20/lib/x64"
)
list(APPEND CMAKE_PREFIX_PATH ${EXTERNAL_LIB_DIRS})


set(EXTERNAL_LIB_INCLUDES
    "C:/Users/lucciano/Workspace/Libs/bin/glew-2.1.0-win32/glew-2.1.0/include"
    "C:/Users/lucciano/Workspace/Libs/bin/SDL2-devel-2.0.20-VC/SDL2-2.0.20/include"
    "C:/Users/lucciano/Workspace/Libs/bin/eigen"
)
#list(APPEND CMAKE_INCLUDE_PATH ${EXTERNAL_LIB_INCLUDES})

find_package(OpenGL REQUIRED)

find_library(SDL2_LIB "SDL2")
find_library(GLEW_LIB "glew32")
find_library(OPENGL_LIB "opengl")
#find_library(CUDART_LIB "cudart")

#FindCUDAToolkit(11.6 REQUIRED)
#find_package(OPENGL 3.4 REQUIRED)
#find_package(CUDA 11.6 REQUIRED)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

target_link_libraries(cuda_opengl_app PRIVATE "${SDL2_LIB}")
target_link_libraries(cuda_opengl_app PRIVATE "${GLEW_LIB}")
target_link_libraries(cuda_opengl_app PRIVATE opengl32)

#target_link_libraries(cuda_opengl_app CUDA::cudart)
#target_link_libraries(cuda_opengl_app PRIVATE "${CUDART_LIB}")
target_include_directories(cuda_opengl_app PUBLIC ${EXTERNAL_LIB_INCLUDES})

return()