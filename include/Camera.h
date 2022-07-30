#ifndef CAMERA_H
#define CAMERA_H

#include "Utils/Eigen.h"
#include "Utils/Math.h"

struct Camera
{
	enum Actions{
		ORBIT_LEFT,
		ORBIT_RIGHT,
		ORBIT_UP,
		ORBIT_DOWN,
		ZOOM_IN,
		ZOOM_OUT,
		MOVE_X_P,
		MOVE_X_M,
		MOVE_Y_P,
		MOVE_Y_M,
		MOVE_Z_P,
		MOVE_Z_M,
	};

	Camera(
		float _windowWidth,
		float _windowHeight,
		ei::Vector3f const & _eye={0.f, 0.f, 30.f},
    	ei::Vector3f const & _target={0.f, 0.f, 0.f},
		float _fov = TO_RAD(50), //In rads 
		float _far = 200.f,
		float _near = 1.f,
		float _rotationSpeed = 0.05f,
		float _zoomSpeed = 0.05f,
		float _xFormSpeed = 1.0f);

	ei::Matrix4f viewMat;

	ei::Matrix4f projMat;

	//Perspective Parameters
	float windowWidth, windowHeight, fov, far, near;

	//Trackball parameters
	float yaw, pitch, zoom;

	float rotationSpeed, zoomSpeed, xFormSpeed;

	ei::Vector3f target, eye, transformedEye;
};

void setProjMat(
	ei::Matrix4f & projMat,
	float windowWidth, 
    float windowHeight, 
    float fov, 
    float far,
	float near);
void updateProjMat(Camera & camera);

void setLookAt(
	ei::Matrix4f & viewMat,
	ei::Vector3f const & position,
	ei::Vector3f const & target,
	ei::Vector3f const & up);
void updateLookAt(Camera & camera);

void yawCamera(Camera& camera, float yawAngle);
void pitchCamera(Camera& camera, float pitchAngle);
void zoomCamera(Camera& camera, float zoomAmount);
void translateCamera(Camera& camera, ei::Vector3f& xForm);
void moveCamera(Camera& camera, Camera::Actions action);

void updateCamera(Camera& camera);

#endif /* CAMERA_H */
