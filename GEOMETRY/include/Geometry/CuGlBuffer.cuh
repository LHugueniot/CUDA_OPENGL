#ifndef CU_GL_BUFFER_CUH
#define CU_GL_BUFFER_CUH

#include <iostream>
#include <sstream>
#include <vector>

#include "Utils/CudaGL.cuh"

template <typename T>
struct cugl_buffer
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
bool allocate_cugl_buffer(cugl_buffer<T> *buffer_obj)
{
    glGenBuffers(1, &buffer_obj->gl_VBO);
    std::cout << "gl_vbo: " << buffer_obj->gl_VBO << std::endl;
    assert(buffer_obj->gl_VBO > 0);
    glBindBuffer(GL_ARRAY_BUFFER, buffer_obj->gl_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(T) * buffer_obj->d_bufferSize, 0,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&buffer_obj->cugl_pVBO, buffer_obj->gl_VBO,
                                 cudaGraphicsMapFlagsWriteDiscard);
    return true;
}

template <typename T>
bool set_cugl_buffer(cugl_buffer<T> *buffer_obj, T *h_pBuffer,
                     size_t h_bufferSize)
{
    std::cout << "Test" << std::endl;
    std::ostringstream oss;

    oss << "h_bufferSize: " << h_bufferSize
        << " buffer_obj->d_bufferSize:" << buffer_obj->d_bufferSize << std::endl;

    ASSERT_WITH_MESSAGE(h_bufferSize == buffer_obj->d_bufferSize, oss.str());

    size_t bufferSize;

    // Map buffer object
    cudaGraphicsMapResources(1, &buffer_obj->cugl_pVBO, 0);

    // Get pointer to use, not sure if possible to use outside of mapped scope
    cudaGraphicsResourceGetMappedPointer((void **)&buffer_obj->d_pBuffer, &bufferSize,
                                         buffer_obj->cugl_pVBO);

    // Copy copy data from host to device buffer
    cudaMemcpy(buffer_obj->d_pBuffer, h_pBuffer, bufferSize, cudaMemcpyHostToDevice);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &buffer_obj->cugl_pVBO, 0);

    return true;
}

template <typename T>
bool set_cugl_buffer(cugl_buffer<T> *buffer_obj, std::vector<T> *h_buffer)
{

    assert(h_buffer->size() == buffer_obj->d_bufferSize);

    auto &d_pBuffer = buffer_obj->d_pBuffer;
    auto &cugl_pVBO = buffer_obj->cugl_pVBO;
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