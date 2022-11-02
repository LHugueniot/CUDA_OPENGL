#include "Utils/Eigen.h"

namespace ei_utils{

void setProjMat(ei::Matrix4f & projMat, float windowWidth, 
    float windowHeight, float fov, float far, float near)
{
    projMat.setIdentity();
    float aspect = float(windowWidth)/float(windowHeight);
    float theta = fov * .5f;
    float range = far - near;
    float invtan = 1./tan(theta);

    projMat(0,0) = invtan / aspect;
    projMat(1,1) = invtan;
    projMat(2,2) = -(near + far) / range;
    projMat(3,2) = -1;
    projMat(2,3) = -2 * near * far / range;
    projMat(3,3) = 0;

    //projMat.transposeInPlace();
}

void setLookAt(ei::Matrix4f & viewMat, ei::Vector3f const & position,
    ei::Vector3f const & target, ei::Vector3f const & up)
{
    viewMat.setZero();

    ei::Matrix3f R;
    R.col(2) = (position-target).normalized();
    R.col(0) = up.cross(R.col(2)).normalized();
    R.col(1) = R.col(2).cross(R.col(0));
    viewMat.topLeftCorner<3, 3>() = R.transpose();
    viewMat.topRightCorner<3, 1>() = -R.transpose() * position;
    viewMat(3, 3) = 1.0f;

    //viewMat.transposeInPlace();
}

} /* END NAMESPACE ei_utils */