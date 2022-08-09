#ifndef UTIL_EIGEN_H
#define UTIL_EIGEN_H

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace Eigen
{
typedef Matrix<unsigned int,2,1> Vector2ui;
typedef Matrix<unsigned int,3,1> Vector3ui;
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
