#include "Utils/General.h"

template <typename T>
__host__ __device__ void setBoolFromPackedBuffer(T *buffer, uint nBufferElems, uint bufferIdx, bool val)
{
    size_t typeSize = sizeof(T);
    size_t typeSizeInBits = (typeSize * BYTE_BITS);

    uint bufferElemIndex = bufferIdx / typeSizeInBits;

#ifdef DEBUG
#ifndef __CUDA__ARCH__
    assert(!(bufferElemIndex > nBufferElems));
#endif
#endif

    uint bufferElemBitOffset = bufferIdx % typeSizeInBits;
    T &bufferElem = buffer[bufferElemIndex];
    bufferElem |= std::byte(val << bufferElemBitOffset);
}

template <typename T>
__host__ __device__ bool getBoolFromPackedBuffer(T *buffer, uint nBufferElems, uint bufferIdx)
{
    size_t typeSize = sizeof(T);
    size_t typeSizeInBits = (typeSize * BYTE_BITS);

    uint bufferElemIndex = bufferIdx / typeSizeInBits;

#ifdef DEBUG
#ifndef __CUDA__ARCH__
    assert(!(bufferElemIndex > nBufferElems));
#endif
#endif

    uint bufferElemBitOffset = bufferIdx % typeSizeInBits;
    T &bufferElem = buffer[bufferElemIndex];
    return bool(bufferElem & std::byte(1 << bufferElemBitOffset));
}
