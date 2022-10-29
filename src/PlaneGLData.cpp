#include <vector>

#include "Utils/Eigen.h"
#include "Utils/OpenGL.h"

#include "PlaneGLData.h"


void initPlaneVAO(PlaneGLData & glData){

    glData.verticesSize = glData.vertices->size();
    glGenBuffers(1, &glData.verticesBufferObject);
    updatePlaneVBO(glData);

    glGenVertexArrays(1, &glData.verticesArrayObject);
    glBindVertexArray(glData.verticesArrayObject);

    glBindBuffer(GL_ARRAY_BUFFER, glData.verticesBufferObject);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glBindVertexArray(0);
    glDisableVertexAttribArray(0);
}

void updatePlaneVBO(PlaneGLData const & glData){

    glBindBuffer(GL_ARRAY_BUFFER, glData.verticesBufferObject);
    glBufferData(GL_ARRAY_BUFFER,
        sizeof(GLfloat) * glData.verticesSize,
        glData.vertices->data(), GL_STATIC_DRAW);
}

void drawPlane(PlaneGLData const & glData, Eigen::Matrix4f & cameraMat){

    glUseProgram(*glData.monoColourShader);
    GLuint mvpID = glGetUniformLocation(*glData.monoColourShader, "MVP");
    glUniformMatrix4fv(mvpID, 1, GL_FALSE, cameraMat.data());

    GLuint baseColID = glGetUniformLocation(*glData.monoColourShader, "base_colour");
    glUniform3fv(baseColID, 1, glData.baseColour.data());

    glBindVertexArray(glData.verticesArrayObject);

    glDrawArrays(GL_LINES, 0, glData.verticesSize/3);
}
