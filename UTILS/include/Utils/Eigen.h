#ifndef UTIL_EIGEN_H
#define UTIL_EIGEN_H

#ifdef __CUDACC__
#pragma warning(disable : 4068)
#pragma nv_diag_suppress 20236
#pragma nv_diag_suppress 20014
#pragma nv_diag_suppress 20012
#pragma warning(disable : 4068)
#endif

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Geometry>

#ifdef __CUDACC__
#pragma warning(default : 4068)
#endif

namespace Eigen
{

using Vector2ui = Matrix<unsigned int, 2, 1>;
using Vector3ui = Matrix<unsigned int, 3, 1>;
// using Scale3f = DiagonalMatrix<Scalar,3>;
using Transform3f = Transform<float, 3, Affine>;
using Translation3f = Translation<float, 3>;

} // namespace Eigen

namespace ei = Eigen;

namespace ei_utils
{

void setProjMat(ei::Matrix4f &projMat, float windowWidth, float windowHeight,
                float fov, float far, float near);

void setLookAt(ei::Matrix4f &viewMat, ei::Vector3f const &position,
               ei::Vector3f const &target, ei::Vector3f const &up);

/*
__host__ float length(ei::Vector3f const &v)
{
    return sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
}
*/

} // namespace ei_utils

#endif /* UTIL_EIGEN_H */
