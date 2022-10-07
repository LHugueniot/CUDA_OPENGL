#include "Utils/Eigen.h"
#include "Utils/Math.h"

#include "Camera.h"
#include <iostream>

Camera::Camera(
    float _windowWidth, float _windowHeight,
    ei::Vector3f const & _eye,
    ei::Vector3f const & _target, 
    float _fov, float _far, float _near,
    float _rotationSpeed, float _zoomSpeed,
    float _xFormSpeed
    ) :
        windowWidth(_windowWidth),
        windowHeight(_windowHeight),
        eye(_eye),
        target(_target),
        fov(_fov),
        far(_far), near(_near),
        rotationSpeed(_rotationSpeed),
        zoomSpeed(_zoomSpeed),
        xFormSpeed(_xFormSpeed)
{
    std::cout<<"eye: "<<_eye<<std::endl;
    std::cout<<"target: "<<_target<<std::endl;
    transformedEye = eye;
    yaw = 0.f;
    pitch = 0.f;
    zoom = 1.f;

    //setProjMat(projMat, windowWidth, windowHeight, fov, far, near);
    updateProjMat(*this);
}

void updateProjMat(Camera & camera)
{
    ei_utils::setProjMat(camera.projMat, 
        camera.windowWidth,
        camera.windowHeight,
        camera.fov,
        camera.far,
        camera.near);
}

void yawCamera(Camera& camera, float yawAngle)
{
    camera.yaw -= yawAngle;

    if (camera.yaw > M_PI)
        camera.yaw -= 2.0 * M_PI;
    else if (camera.yaw < -M_PI)
        camera.yaw += 2.0 * M_PI;
}

void pitchCamera(Camera& camera, float pitchAngle)
{
    camera.pitch = std::clamp(camera.pitch + pitchAngle,
        -.5f * (float)M_PI, .5f * (float)M_PI);
}

void zoomCamera(Camera& camera, float zoomAmount)
{
    camera.zoom = std::clamp(camera.zoom + zoomAmount, 0.f, 10.f);
}

void translateCamera(Camera& camera, ei::Vector3f const& xForm)
{
    camera.eye += xForm;
    camera.target += xForm;
}

void moveCamera(Camera& camera, Camera::Actions action)
{
    switch (action)
    {
        case Camera::ORBIT_LEFT:
            yawCamera(camera, camera.rotationSpeed);
            break;
        case Camera::ORBIT_RIGHT:
            yawCamera(camera, -camera.rotationSpeed);
            break;
        case Camera::ORBIT_UP:
            pitchCamera(camera, -camera.rotationSpeed);
            break;
        case Camera::ORBIT_DOWN:
            pitchCamera(camera, camera.rotationSpeed);
            break;
        case Camera::ZOOM_IN:
            zoomCamera(camera, camera.zoomSpeed);
            break;
        case Camera::ZOOM_OUT:
            zoomCamera(camera, -camera.zoomSpeed);
            break;
        case Camera::MOVE_X_P:
            translateCamera(camera, {camera.xFormSpeed, 0, 0});
            break;
        case Camera::MOVE_X_M:
            translateCamera(camera, {-camera.xFormSpeed, 0, 0});
            break;
        case Camera::MOVE_Y_P:
            translateCamera(camera, {0, camera.xFormSpeed, 0});
            break;
        case Camera::MOVE_Y_M:
            translateCamera(camera, {0, -camera.xFormSpeed, 0});
            break;
        case Camera::MOVE_Z_P:
            translateCamera(camera, {0, 0, camera.xFormSpeed});
            break;
        case Camera::MOVE_Z_M:
            translateCamera(camera, {0, 0, -camera.xFormSpeed});
            break;
    }
}

void updateCamera(Camera& camera)
{
    ei::Matrix3f R_yaw; 
    R_yaw = ei::AngleAxisf(camera.yaw, ei::Vector3f::UnitY());
    ei::Matrix3f R_pitch;
    R_pitch = ei::AngleAxisf(camera.pitch, ei::Vector3f::UnitX());
    camera.transformedEye = (R_yaw * R_pitch * (
        camera.zoom * (camera.eye - camera.target))) + camera.target;
    ei_utils::setLookAt(camera.viewMat, camera.transformedEye,
        camera.target, ei::Vector3f::UnitY());
}
