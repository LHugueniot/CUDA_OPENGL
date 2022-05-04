#ifndef UTIL_CUDA_CUH
#define UTIL_CUDA_CUH

// Cuda includes (tb expanded)
#include <cuda_runtime.h>

#define cutilSafeCall(err)  __cudaSafeCall(err,__FILE__,__LINE__)
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
    if(cudaSuccess != err) {
      printf("%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
             file, line, cudaGetErrorString(err) );
      exit(-1);
    }
}

#endif /* UTIL_CUDA_CUH */