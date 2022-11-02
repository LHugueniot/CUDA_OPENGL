#ifndef CU_GL_GEOMETRY_CUH
#define CU_GL_GEOMETRY_CUH

#include <vector>

#include "Utils/Eigen.h"
#include "Utils/General.h"

#include "Geometry/CuGlBuffer.cuh"

void __global__ setBufferVals(float setNum, float *d_bufferPtr, int bufferSize);

void __global__ addToBufferVertex(ei::Vector3f setNum, float *d_bufferPtr,
                                  int bufferSize);

struct CuGlGeometry
{
    CuGlGeometry(float *h_pBuffer,     // Host side buffer data
                 size_t bufferElemNum, // Number of buffer elements
                 GLuint *_monoColourShader)
        : // Shader id
          monoColourShader(_monoColourShader)
    {
        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);
        buffer.d_bufferSize = bufferElemNum;
        allocate_cugl_buffer(&buffer);
        set_cugl_buffer(&buffer, h_pBuffer, bufferElemNum);
    }
    CuGlGeometry(std::vector<float> *h_Buffer, // Host side buffer data
                 GLuint *_monoColourShader)
        : // Shader id
          monoColourShader(_monoColourShader)
    {
        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);

        glBindBuffer(GL_ARRAY_BUFFER, buffer.gl_VBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

        buffer.d_bufferSize = h_Buffer->size();
        allocate_cugl_buffer(&buffer);
        set_cugl_buffer(&buffer, h_Buffer);
    }
    GLuint m_vao;
    cugl_buffer<float> buffer = {};

    GLuint *monoColourShader;

    ei::Vector3f baseColour = {1, 0, 0};
};

void drawGeom(CuGlGeometry const &geom, Eigen::Matrix4f &cameraMat);

void translateGeom(CuGlGeometry &geom, const ei::Vector3f &setNum);

#endif /* CU_GL_GEOMETRY_CUH */
