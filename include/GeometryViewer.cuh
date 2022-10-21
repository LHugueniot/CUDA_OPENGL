#ifndef GEOMETRY_VIEWER_CUH
#define GEOMETRY_VIEWER_CUH

#include <vector>

#include "Utils/Eigen.h"
#include "Utils/General.h"
#include "Utils/OpenGL.h"

struct GeometryViewer
{
    GLuint m_vao = 0;

    uint m_nVertices = 0;
    GLuint m_vbo = 0;

    uint m_nIndices = 0;
    GLuint m_ibo = 0;

    GLuint *m_monoColourShader = nullptr;

    ei::Vector3f m_baseColour = {1, 0, 0};
};

void initGeometryViewer(GeometryViewer &geom, uint nVertices,
                        GLuint vbo, uint nIndices, GLuint ibo,
                        GLuint *monoColorShader,
                        const ei::Vector3f &baseColour = {1, 0, 0});
void drawGeometryViewer(GeometryViewer const &geom, Eigen::Matrix4f &cameraMat);

#endif /* GEOMETRY_VIEWER_CUH */
