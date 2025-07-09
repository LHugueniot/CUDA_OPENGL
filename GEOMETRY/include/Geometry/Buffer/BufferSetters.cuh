#ifndef BUFFER__BUFFER_SETTERS_CUH
#define BUFFER__BUFFER_SETTERS_CUH

#include "Utils/CudaGL.cuh"

#include <cassert>

// WTF is this Lucien?
template <typename T, GLenum GLBufferType = GL_ARRAY_BUFFER>
struct CuGlBufferSetter
{
    void allocate(T **devPtr, size_t nElems)
    {
        m_nElements = nElems;
        m_sizeOfElement = sizeof(T);

        size_t bufferSize = m_nElements * m_sizeOfElement;

        glGenBuffers(1, &m_glBufferId);
        checkGLError();
        glBindBuffer(GLBufferType, m_glBufferId);
        checkGLError();
        glBufferData(GLBufferType, m_nElements * m_sizeOfElement, nullptr,
                     GL_DYNAMIC_DRAW);
        checkGLError();

        // Map buffer object
        cutilSafeCall(cudaGraphicsGLRegisterBuffer(&m_resourceObj, m_glBufferId,
                                                   cudaGraphicsMapFlagsWriteDiscard));

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

        glBindBuffer(GLBufferType, 0);
        checkGLError();
    }

    void mapAndSync(T **devPtr)
    {
        // Map buffer object
        cutilSafeCall(cudaGraphicsMapResources(1, &m_resourceObj, 0));

        size_t size;
        // Get pointer to use, not sure if possible to use outside of mapped scope
        cutilSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void **>(devPtr), &size, m_resourceObj));
        //std::cout << "size: " << size << std::endl;
    }

    void unMap()
    {
        // Map buffer object
        cutilSafeCall(cudaGraphicsUnmapResources(1, &m_resourceObj, 0));
    }

    std::vector<T> m_data = {};

    GLuint m_glBufferId = 0;

    size_t m_nElements = 0;
    size_t m_sizeOfElement = 0;

    struct cudaGraphicsResource *m_resourceObj = nullptr;
};

template <typename T>
struct DefaultCudaBufferSetter
{
    void allocate(T **devPtr, size_t nElems)
    {
        cutilSafeCall(
            cudaMalloc(reinterpret_cast<void **>(devPtr), nElems * sizeof(T)));
    }

    void copy(T *devPtr, T *data, size_t nElems)
    {
        size_t bufferSize = nElems * sizeof(T);
        m_data.resize(nElems);
        memcpy(&m_data.data()[0], data, bufferSize);
        cutilSafeCall(cudaMemcpy(devPtr, data, bufferSize, cudaMemcpyHostToDevice));
    }
    std::vector<T> m_data = {};
};


#endif /* BUFFER__BUFFER_SETTERS_CUH */
