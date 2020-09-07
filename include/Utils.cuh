#ifndef UTILITIES_H
#define UTILITIES_H

// c++11 std lib includes
#include <map>
#include <cmath>
#include <string>
#include <memory>
#include <vector>
#include <random>
#include <chrono>
#include <sstream>
#include <cassert>
#include <iostream>
#include <algorithm>

namespace std {
	template<class T>
	constexpr const T& clamp( const T& v, const T& lo, const T& hi ) {
	    assert( !(hi < lo) );
	    return (v < lo) ? lo : (hi < v) ? hi : v;
	}
}

// Window management
#include <SDL2/SDL.h>

// Graphics drawing
#include <GL/glew.h>

// Cuda includes (tb expanded)
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Eigen includes
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

#define M_RAD_RATIO 0.01745329251
#define TO_RAD(deg) deg * M_RAD_RATIO

#define cot(x) cos(x)/sin(x)

#define cutilSafeCall(err)  __cudaSafeCall(err,__FILE__,__LINE__)
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
    if(cudaSuccess != err) {
      printf("%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
             file, line, cudaGetErrorString(err) );
      exit(-1);
    }
}



#define ASSERT_WITH_MESSAGE(condition, message) do { \
if (!(condition)) { std::cout<<message; } \
assert ((condition)); } while(false)



#endif /* UTILITIES_H */