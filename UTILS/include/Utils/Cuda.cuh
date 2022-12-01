#ifndef UTIL_CUDA_CUH
#define UTIL_CUDA_CUH

#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
// Cuda includes (tb expanded)
#include <cuda_runtime.h>

#define cutilSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError err, const std::string_view file, const int line)
{
    if (cudaSuccess != err)
    {
        std::stringstream errStream;
        errStream << file << "(" << std::to_string(line) << ") : ";
        errStream << "cutilSafeCall() Runtime API error : " << cudaGetErrorString(err) << ".";

        std::string errMsg = errStream.str();
        std::cout << errMsg << std::endl;

        throw std::runtime_error(errMsg);
    }
}

#endif /* UTIL_CUDA_CUH */
