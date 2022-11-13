#include <type_traits>

#include <thrust/device_ptr>

#include "Utils/CudaGL.cuh"

template <typename T>
static constexpr GLenum getGlEnumPODForHostPOD(T)
{
    if (std::is_same<T, char>::value)
        return GL_BYTE;
    else if (std::is_same<T, unsigned char>::value)
        return GL_UNSIGNED_BYTE;
    else if (std::is_same<T, short int>::value)
        return GL_SHORT;
    else if (std::is_same<T, unsigned short int>::value)
        return GL_UNSIGNED_SHORT;
    else if (std::is_same<T, int>::value)
        return GL_INT;
    else if (std::is_same<T, unsigned int>::value)
        return GL_UNSIGNED_INT;
    else if (std::is_same<T, short float>::value)
        return GL_HALF_FLOAT;
    else if (std::is_same<T, float>::value)
        return GL_FLOAT;
    else if (std::is_same<T, double>::value)
        return GL_DOUBLE;
    else
    {
        auto errMsg = std::string("Invalid template argument: ") + std::string(typeid(T).name());
        throw std::invalid_argument(errMsg);
    }
}

template <typename PODType>
struct CuGlInfo
{
    CuGlInfo(GLenum _glPODType = getGlEnumPODForHostPOD<PODType>());
    // Opengl vertex buffer object, use to draw
    GLuint m_glBufferId = 0;
    const GLenum m_glPODType = 0;
    GLenum m_glBufferTarget = GL_ARRAY_BUFFER;

    // Cuda/Opengl vertex buffer object pointer (to be expanded on)
    struct cudaGraphicsResource *m_resourceObj = nullptr;

    // Pointer to actual device buffer data
    thrust::device_ptr<PODType> *d_buffer = nullptr;

    // Number of elements in device buffer
    size_t m_bufferCount = 0;
}