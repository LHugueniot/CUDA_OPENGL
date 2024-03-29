#ifndef UTIL_MATH_H
#define UTIL_MATH_H

//#define _USE_MATH_DEFINES
#include <cmath>

#define RAD_RATIO 0.01745329251f
#define TO_RAD(deg) deg *RAD_RATIO

#define cot(x) cos(x) / sin(x)

enum Dim
{
    X = 0,
    Y = 1,
    Z = 2
};

#endif /* UTIL_MATH_H */