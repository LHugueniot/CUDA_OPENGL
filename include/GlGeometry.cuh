#ifndef GL_GEOMETRY_CUH
#define GL_GEOMETRY_CUH

#include <iostream>
#include <sstream>
#include <vector>

#include "Geometry.cuh"
#include "Utils/CudaGL.cuh"

template <typename T, GLenum GLBufferType = GL_ARRAY_BUFFER>
struct CuGlBufferSetter
{
    void allocate(T **devPtr, size_t nElems)
    {
        m_nElements = nElems;
        m_sizeOfElement = sizeof(T);
        glGenBuffers(1, &m_glBufferId);
        glBindBuffer(GLBufferType, m_glBufferId);
        glBufferData(GLBufferType, m_nElements * m_sizeOfElement, 0,
                     GL_DYNAMIC_DRAW);
        glBindBuffer(GLBufferType, 0);

        // Map buffer object
        cutilSafeCall(cudaGraphicsGLRegisterBuffer(&m_resourceObj, m_glBufferId,
                                                   cudaGraphicsRegisterFlagsNone));

        // Map buffer object
        cutilSafeCall(cudaGraphicsMapResources(1, &m_resourceObj, 0));

        size_t size;
        // Get pointer to use, not sure if possible to use outside of mapped scope
        cutilSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void **>(devPtr), &size, m_resourceObj));
        assert(size == (m_nElements * m_sizeOfElement));
    }

    void copy(T *devPtr, T *data, size_t nElems)
    {
        size_t bufferSize = nElems * sizeof(T);
        m_data.resize(nElems);
        memcpy(&m_data.data()[0], data, bufferSize);

        // Copy copy data from host to device buffer
        cutilSafeCall(
            cudaMemcpy(devPtr, data, nElems * sizeof(T), cudaMemcpyHostToDevice));
        // Unmap buffer object
        cudaGraphicsUnmapResources(1, &m_resourceObj, 0);
    }

    std::vector<T> m_data;

    GLuint m_glBufferId;

    size_t m_nElements;
    size_t m_sizeOfElement;

    struct cudaGraphicsResource *m_resourceObj;
};

#endif /* GL_GEOMETRY_CUH */
