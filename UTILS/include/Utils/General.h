#ifndef UTIL_GENERAL_H
#define UTIL_GENERAL_H

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <iostream>
#include <limits.h>
#include <list>
#include <map>
#include <set>
#include <vector>
#include<stdio.h>

//#ifndef __cplusplus < 201703L
// namespace std {
//    template<class T>
//    constexpr const T& clamp( const T& v, const T& lo, const T& hi ) {
//        assert( !(hi < lo) );
//        return (v < lo) ? lo : (hi < v) ? hi : v;
//    }
//}
//#endif

// Type Defines

using std::byte;

#ifndef uint
using uint = unsigned int;
#endif

#define BYTE_BITS 8

// Useful macros

#define ASSERT_WITH_MESSAGE(condition, message) \
    do                                          \
    {                                           \
        if (!(condition))                       \
        {                                       \
            std::cout << message;               \
        }                                       \
        assert((condition));                    \
    } while (false)

#define WARNING_MSG(msg) printf("BREAK POINT - FILE %s - LINE %i : %s\n", __FILE__, __LINE__, msg)

// Std container print outs

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

std::pair<uint, uint> makeOrderedIdxPair(uint idx1, uint idx2);

struct TabIndentContext
{
    TabIndentContext();

    ~TabIndentContext();

    friend std::ostream &operator<<(std::ostream &os, const TabIndentContext &ctx);
};

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, const std::pair<T1, T2> &idxPair)
{
    os << "{" << idxPair.first << ", " << idxPair.second << "}";
    return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::list<T> &l)
{
    TabIndentContext tabCtx;

    os << "{\n";
    for (auto &e : l)
    {
        os << tabCtx << e << ",\n";
    }
    os << "}";
    return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::set<T> &s)
{
    TabIndentContext tabCtx;

    os << "{\n";
    for (auto &e : s)
    {
        os << tabCtx << e << ",\n";
    }
    os << "}";
    return os;
}
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << "{\n";
    {
        TabIndentContext tabCtx;

        for (auto &e : v)
        {
            os << tabCtx << e << ",\n";
        }
    }
    os << "}";
    return os;
}

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, const std::map<T1, T2> &m)
{
    TabIndentContext tabCtx;
    os << tabCtx << "{\n";
    for (auto &[k, v] : m)
    {
        TabIndentContext tabCtx2;
        os << tabCtx2 << k << " :\n";

        TabIndentContext tabCtx3;
        os << tabCtx3 << v << ",\n";
    }
    os << tabCtx << "\n}\n";
    return os;
}

#endif /* UTIL_GENERAL_H */
