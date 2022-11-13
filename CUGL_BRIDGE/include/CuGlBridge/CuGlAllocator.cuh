#include <type_traits>

#include <thrust/device_vector.h>

#include "Utils/CudaGL.cuh"
template <typename PODType>
class gl_device_allocator : thrust::device_allocator<PODType>
{

}

template <typename PODType, typename Alloc = gl_device_allocator<PODType>>
class gl_device_vector : thrust::device_vector
{

    // ================== Cuda GL interop data=================================

    // Opengl vertex buffer object, use to draw
    GLuint m_glBufferId = 0;
    const GLenum m_glPODType = 0;
    GLenum m_glBufferTarget = GL_ARRAY_BUFFER;

    // Cuda/Opengl vertex buffer object pointer (to be expanded on)
    struct cudaGraphicsResource *m_resourceObj = nullptr;

    // Pointer to actual device buffer data
    thrust::device_ptr<PODType> *d_buffer = nullptr;

    // Number of elements in device buffer
    size_t bufferSize = 0;
}