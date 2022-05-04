#ifndef UTIL_GENERAL_H
#define UTIL_GENERAL_H

#include <algorithm>

#ifndef __cplusplus < 201703L
namespace std {
	template<class T>
	constexpr const T& clamp( const T& v, const T& lo, const T& hi ) {
	    assert( !(hi < lo) );
	    return (v < lo) ? lo : (hi < v) ? hi : v;
	}
}
#endif

#ifndef uint
using uint=unsigned int;
#endif

#define ASSERT_WITH_MESSAGE(condition, message) do { \
if (!(condition)) { std::cout<<message; } \
assert ((condition)); } while(false)

#endif /* UTIL_GENERAL_H */