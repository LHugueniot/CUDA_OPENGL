#ifndef UTIL_GENERAL_H
#define UTIL_GENERAL_H

#include <algorithm>
#include <iostream>
#include <vector>

//#ifndef __cplusplus < 201703L
// namespace std {
//    template<class T>
//    constexpr const T& clamp( const T& v, const T& lo, const T& hi ) {
//        assert( !(hi < lo) );
//        return (v < lo) ? lo : (hi < v) ? hi : v;
//    }
//}
//#endif

#ifndef uint
using uint = unsigned int;
#endif

#define ASSERT_WITH_MESSAGE(condition, message) \
    do                                          \
    {                                           \
        if (!(condition))                       \
        {                                       \
            std::cout << message;               \
        }                                       \
        assert((condition));                    \
    } while (false)

template <typename T>
void printStdVecInStride(std::vector<T> &vec, uint stride = 3)
{
    for (size_t i = 0; i < vec.size(); i += 3)
    {
        for (size_t j = 0; (j < stride) && (i + j < vec.size()); j++)
        {
            std::cout << vec[i + j] << ", ";
        }
        std::cout << std::endl;
    }
}

#endif /* UTIL_GENERAL_H */
