#include "Utils/Eigen.h"
#include "Utils/Math.h"

#include "Camera.h"

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
        xFormSpeed(_xFormSpeed){

    transformedEye = eye;
    yaw = 0.f;
    pitch = 0.f;
    zoom = 1.f;

    setProjMat(projMat, windowWidth, windowHeight, fov, far, near);
    //updateCamera(*this);
}

void setProjMat(ei::Matrix4f & projMat, float windowWidth, 
    float windowHeight, float fov, float far, float near){

    projMat.setIdentity();
    float aspect = float(windowWidth)/float(windowHeight);
    float theta = fov * .5f;
    float range = far - near;
    float invtan = 1./tan(theta);

    projMat <<  invtan/aspect, 0, 0, 0,
                0, invtan, 0, 0,
                0, 0, -(near+far)/range, -1,
                0, 0, -2*near*far/range, 0;

    //projMat(0,0) = invtan / aspect;
    //projMat(1,1) = invtan;
    //projMat(2,2) = -(near + far) / range;
    //projMat(3,2) = -1;
    //projMat(2,3) = -2 * near * far / range;
    //projMat(3,3) = 0;
}

void updateProjMat(Camera & camera){
    setProjMat(camera.projMat, 
        camera.windowWidth,
        camera.windowHeight,
        camera.fov,
        camera.far,
        camera.near);
}

void rotateCamera(Camera& camera, float rotateAngle)
{
    camera.yaw -= rotateAngle;

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
            rotateCamera(camera, camera.rotationSpeed);
            break;
        case Camera::ORBIT_RIGHT:
            rotateCamera(camera, -camera.rotationSpeed);
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

void setLookAt(ei::Matrix4f & viewMat, ei::Vector3f const & position,
    ei::Vector3f const & target, ei::Vector3f const & up)
{
    //ei::Vector3f direction = position != target ?
    //                         position-target : 
    //                         ei::Vector3f(1, 0, 0);

    ei::Matrix3f R;
    R.col(2) = (position-target).normalized();
    R.col(0) = up.cross(R.col(2)).normalized();
    R.col(1) = R.col(2).cross(R.col(0));
    viewMat.topLeftCorner<3, 3>() = R.transpose();
    viewMat.topRightCorner<3, 1>() = -R.transpose() * position;
    viewMat(3, 3) = 1.0f;
}

void updateCamera(Camera& camera){

    ei::Matrix3f R_yaw; 
    R_yaw = ei::AngleAxisf(camera.yaw, ei::Vector3f::UnitY());
    ei::Matrix3f R_pitch;
    R_pitch = ei::AngleAxisf(camera.pitch, ei::Vector3f::UnitX());
    camera.transformedEye = (R_yaw * R_pitch * (
        camera.zoom * (camera.eye - camera.target))) + camera.target;
    setLookAt(camera.viewMat, camera.transformedEye,
        camera.target, ei::Vector3f(0, 1, 0));
}

/*

void updateCamera(Camera& camera){

    glm::dmat3 R_yaw = glm::mat3_cast(glm::angleAxis(camera.yaw, ei::Vector3f(0.0, 1.0, 0.0)));
    glm::dmat3 R_pitch = glm::mat3_cast(glm::angleAxis(camera.pitch, ei::Vector3f(1.0, 0.0, 0.0)));
    camera.transformedEye = (R_yaw * R_pitch * (camera.zoom * (camera.eye-camera.target))) + camera.target;
    camera.viewMat = glm::lookAt(glm::vec3(camera.transformedEye), glm::vec3(camera.target), glm::vec3(0.0f,1.0f,0.0f));
}

void moveCamera(Camera& camera, cameraActions action)
{

	
    float rotationSpeed = 0.05f;

	switch (action)
	{
		case ORBIT_LEFT:
			camera.translationMat = glm::rotate(glm::mat4(1.0f), rotationSpeed, glm::vec3(0.0f, 1.0f, 0.0f)) * camera.translationMat; 
			break;
		case ORBIT_RIGHT:
			camera.translationMat = glm::rotate(glm::mat4(1.0f), -rotationSpeed, glm::vec3(0.0f, 1.0f, 0.0f)) * camera.translationMat; 
			break;
		case PAN_RIGHT:
			camera.pivotPointMat = glm::translate(glm::mat4(1.0f), glm::vec3(0.1f, 0.0f,0.0f)) * camera.pivotPointMat;
			break;
		case PAN_LEFT:
			camera.pivotPointMat = glm::translate(glm::mat4(1.0f), glm::vec3(-0.1f, 0.0f,0.0f)) * camera.pivotPointMat;
			break;
		case ZOOM_IN:
			camera.translationMat = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f,0.1f)) * camera.translationMat;
			break;
		case ZOOM_OUT:
			camera.translationMat = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f,-0.1f)) * camera.translationMat;
			break;
		case FORWARD:
			camera.pivotPointMat = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f,0.1f)) * camera.pivotPointMat;
			break;
		case BACK:
			camera.pivotPointMat = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f,-0.1f)) * camera.pivotPointMat;
			break;
	}

    glm::dmat3 R_yaw = glm::mat3_cast(glm::angleAxis(camera.yaw, ei::Vector3f(0.0, 1.0, 0.0)));
    glm::dmat3 R_pitch = glm::mat3_cast(glm::angleAxis(m_pitch, ei::Vector3f(1.0, 0.0, 0.0)));
    m_transformedEye = (R_yaw * R_pitch * (m_zoom * (m_eye-m_target))) + m_target;
    m_V = glm::lookAt(glm::vec3(m_transformedEye), glm::vec3(m_target), glm::vec3(0.0f,1.0f,0.0f));

	camera.viewMat = glm::inverse(camera.translationMat * camera.pivotPointMat );
}
*/

//glm::mat4 defaultcameraMatrix(float width, float height)
//{
//	glm::mat4 Projection = glm::perspective(glm::radians(45.0f), (float) width / (float)height, 0.1f, 100.0f);
//	  
//	// Or, for an ortho camera :
//	//glm::mat4 Projection = glm::ortho(-10.0f,10.0f,-10.0f,10.0f,0.0f,100.0f); // In world coordinates
//	  
//	// cameraMatrix matrix
//	glm::mat4 View = glm::lookAt(
//	    glm::vec3(10,10,10), // cameraMatrix is at (4,3,3), in World Space
//	    glm::vec3(0,0,0), // and looks at the origin
//	    glm::vec3(0,1,0)  // Head is up (set to 0,-1,0 to look upside-down)
//	    );
//	  
//	// Model matrix : an identity matrix (model will be at the origin)
//	glm::mat4 Model = glm::mat4(1.0f);
//
//	//glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.f);
//	//glm::mat4 View = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -Translate));
//	//View = glm::rotate(View, Rotate.y, glm::vec3(-1.0f, 0.0f, 0.0f));
//	//View = glm::rotate(View, Rotate.x, glm::vec3(0.0f, 1.0f, 0.0f));
//	//glm::mat4 Model = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
//	return Projection * View * Model;
//}