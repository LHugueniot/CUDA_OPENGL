#include "Utils/Eigen.h"
#include "Utils/Math.h"

#include "Viewer/Camera.h"
#include <iostream>

Camera::Camera(
    float _windowWidth, float _windowHeight,
    ei::Vector3f const & _eye,
    ei::Vector3f const & _target, 
    float _fov, float _far, float _near,
    float _rotationSpeed, float _zoomSpeed,
    float _xLateSpeed
    ) :
        windowWidth(_windowWidth),
        windowHeight(_windowHeight),
        eye(_eye),
        target(_target),
        fov(_fov),
        far(_far), near(_near),
        rotationSpeed(_rotationSpeed),
        zoomSpeed(_zoomSpeed),
        xLateSpeed(_xLateSpeed)
{
    std::cout<<"eye: "<<_eye<<std::endl;
    std::cout<<"target: "<<_target<<std::endl;
    yaw = 0.f;
    pitch = 0.f;
    zoom = 1.f;
    panTranslation = {0.f, 0.f, 0.f};

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

void panCamera(Camera& camera, float x, float z)
{
    ei::Matrix3f R;
    R = ei::AngleAxisf(camera.yaw, -ei::Vector3f::UnitY());

    ei::Vector3f yawAdjustedXLate = {z, 0.f, x};
    camera.panTranslation -= yawAdjustedXLate.transpose() * R;
    std::cout << "camera.panTranslation:\n{\n" << camera.panTranslation << "\n}" << std::endl;
}

void translateCamera(Camera& camera, ei::Vector3f const& xLate)
{
    camera.eye += xLate;
    camera.target += xLate;
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
            zoomCamera(camera, -camera.zoomSpeed);
            break;
        case Camera::ZOOM_OUT:
            zoomCamera(camera, camera.zoomSpeed);
            break;
        case Camera::MOVE_X_P:
            translateCamera(camera, {camera.xLateSpeed, 0, 0});
            break;
        case Camera::MOVE_X_M:
            translateCamera(camera, {-camera.xLateSpeed, 0, 0});
            break;
        case Camera::MOVE_Y_P:
            translateCamera(camera, {0, camera.xLateSpeed, 0});
            break;
        case Camera::MOVE_Y_M:
            translateCamera(camera, {0, -camera.xLateSpeed, 0});
            break;
        case Camera::MOVE_Z_P:
            translateCamera(camera, {0, 0, camera.xLateSpeed});
            break;
        case Camera::MOVE_Z_M:
            translateCamera(camera, {0, 0, -camera.xLateSpeed});
            break;
        case Camera::PAN_UP:
            panCamera(camera, camera.xLateSpeed, 0.f);
            break;
        case Camera::PAN_DOWN:
            panCamera(camera, -camera.xLateSpeed, 0.f);
            break;
        case Camera::PAN_LEFT:
            panCamera(camera, 0.f, camera.xLateSpeed);
            break;
        case Camera::PAN_RIGHT:
            panCamera(camera, 0.f, -camera.xLateSpeed);
            break;
    }
}

void updateCamera(Camera& camera)
{
    ei::Matrix3f R_yaw; 
    R_yaw = ei::AngleAxisf(camera.yaw, ei::Vector3f::UnitY());

    ei::Matrix3f R_pitch;
    R_pitch = ei::AngleAxisf(camera.pitch, ei::Vector3f::UnitX());

    ei::Vector3f transformedEye = (
        R_yaw * R_pitch * (
            camera.zoom * (
                (camera.eye - camera.target)
            )
        )
    ) + camera.target;

    ei_utils::setLookAt(camera.viewMat, transformedEye + camera.panTranslation,
        camera.target + camera.panTranslation, ei::Vector3f::UnitY());
}
