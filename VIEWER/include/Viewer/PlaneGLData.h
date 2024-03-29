#ifndef PLANE_GL_DATA_H
#define PLANE_GL_DATA_H

#include <vector>

#include "Utils/Eigen.h"
#include "Utils/General.h"
#include "Utils/OpenGL.h"

//=====================================GRID====================================================

struct PlaneGLData
{

    PlaneGLData() {}
    PlaneGLData(std::vector<float> *_vertices,
                GLuint *_monoColourShader,
                ei::Vector3f _baseColour = {1.f, 1.f, 1.f}) : monoColourShader(_monoColourShader),
                                                              baseColour(_baseColour)
    {
        vertices = _vertices;
    }

    GLuint *monoColourShader;

    GLuint verticesSize = 0;
    GLuint verticesBufferObject = 0;
    GLuint verticesArrayObject = 0;

    std::vector<float> *vertices;

    ei::Vector3f baseColour;
};

void initPlaneVAO(PlaneGLData &glData);
void updatePlaneVBO(PlaneGLData const &glData);
void drawPlane(PlaneGLData const &glData, Eigen::Matrix4f &cameraMat);

#endif /* PLANE_GL_DATA_H */