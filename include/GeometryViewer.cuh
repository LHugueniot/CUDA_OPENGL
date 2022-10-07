#ifndef GEOMETRY_VIEWER_CUH
#define GEOMETRY_VIEWER_CUH

#include <vector>

#include "Utils/Eigen.h"
#include "Utils/General.h"
#include "Utils/OpenGL.h"

struct GeometryViewer
{
    GLuint m_vao;

    uint m_nVertices;
    GLuint m_vbo;

    uint m_nIndices;
    GLuint m_ibo;

    GLuint *m_monoColourShader;

    ei::Vector3f m_baseColour = {1, 0, 0};
};

void initGeometryViewer(GeometryViewer &geom, uint nVertices,
                        GLuint vbo, uint nIndices, GLuint ibo,
                        GLuint *monoColorShader,
                        const ei::Vector3f &baseColour = {1, 0, 0});
void drawGeometryViewer(GeometryViewer const &geom, Eigen::Matrix4f &cameraMat);

#endif /* GEOMETRY_VIEWER_CUH */
