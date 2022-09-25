#ifndef UTIL_EIGEN_H
#define UTIL_EIGEN_H

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace Eigen
{
using Vector2ui = Matrix<unsigned int,2,1>;
using Vector3ui = Matrix<unsigned int,3,1>;
//using Scale3f = DiagonalMatrix<Scalar,3>;
using Transform3f = Transform<float, 3, Affine>;
using Translation3f = Translation<float,3>;
}

namespace ei = Eigen;

namespace ei_utils{

void setProjMat(
	ei::Matrix4f & projMat,
	float windowWidth, 
    float windowHeight, 
    float fov, 
    float far,
	float near);

void setLookAt(
	ei::Matrix4f & viewMat,
	ei::Vector3f const & position,
	ei::Vector3f const & target,
	ei::Vector3f const & up);

}


#endif /* UTIL_EIGEN_H */
