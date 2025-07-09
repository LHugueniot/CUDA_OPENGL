#ifndef CU_GL_BUFFER_CUH
#define CU_GL_BUFFER_CUH

#include <iostream>
#include <sstream>
#include <vector>

#include "Utils/CudaGL.cuh"

template <typename T>
struct CuGlBuffer
{
    // Opengl vertex buffer object, use to draw
    GLuint gl_VBO = 0;

    // Cuda/Opengl vertex buffer object pointer (to be expanded on)
    struct cudaGraphicsResource *cugl_pVBO = nullptr;

    // Pointer to actual device buffer data
    T *d_pBuffer = nullptr;

    // Number of elements in device buffer
    size_t d_bufferSize = 0;
};

template <typename T>
bool allocateCuGlBuffer(CuGlBuffer<T> *bufferObj)
{
    glGenBuffers(1, &bufferObj->gl_VBO);
    std::cout << "gl_vbo: " << bufferObj->gl_VBO << std::endl;
    assert(bufferObj->gl_VBO > 0);
    glBindBuffer(GL_ARRAY_BUFFER, bufferObj->gl_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(T) * bufferObj->d_bufferSize, 0,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&bufferObj->cugl_pVBO, bufferObj->gl_VBO,
                                 cudaGraphicsMapFlagsWriteDiscard);
    return true;
}

template <typename T>
bool setCuGlBuffer(CuGlBuffer<T> *bufferObj, T *h_pBuffer,
                     size_t h_bufferSize)
{
    std::cout << "Test" << std::endl;
    std::ostringstream oss;

    oss << "h_bufferSize: " << h_bufferSize
        << " bufferObj->d_bufferSize:" << bufferObj->d_bufferSize << std::endl;

    ASSERT_WITH_MESSAGE(h_bufferSize == bufferObj->d_bufferSize, oss.str());

    size_t bufferSize;

    // Map buffer object
    cudaGraphicsMapResources(1, &bufferObj->cugl_pVBO, 0);

    // Get pointer to use, not sure if possible to use outside of mapped scope
    cudaGraphicsResourceGetMappedPointer((void **)&bufferObj->d_pBuffer, &bufferSize,
                                         bufferObj->cugl_pVBO);

    // Copy copy data from host to device buffer
    cudaMemcpy(bufferObj->d_pBuffer, h_pBuffer, bufferSize, cudaMemcpyHostToDevice);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &bufferObj->cugl_pVBO, 0);

    return true;
}

template <typename T>
bool setCuGlBuffer(CuGlBuffer<T> *bufferObj, std::vector<T> *h_buffer)
{

    assert(h_buffer->size() == bufferObj->d_bufferSize);

    auto &d_pBuffer = bufferObj->d_pBuffer;
    auto &cugl_pVBO = bufferObj->cugl_pVBO;
    size_t bufferSize;

    // Map buffer object
    cudaGraphicsMapResources(1, &cugl_pVBO, 0);

    // Get pointer to use, not sure if possible to use outside of mapped scope
    cudaGraphicsResourceGetMappedPointer((void **)&d_pBuffer, &bufferSize,
                                         cugl_pVBO);

    // Copy copy data from host to device buffer
    cudaMemcpy(d_pBuffer, h_buffer->data(), bufferSize, cudaMemcpyHostToDevice);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &cugl_pVBO, 0);
    return true;
}

#endif /* CU_GL_BUFFER_CUH */