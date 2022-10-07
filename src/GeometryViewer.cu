#include "GeometryViewer.cuh"

void initGeometryViewer(GeometryViewer &geom, uint nVertices, GLuint vbo,
                        uint nIndices, GLuint ibo, GLuint *monoColorShader,
                        const ei::Vector3f &baseColour)
{
    geom.m_nVertices = nVertices;
    geom.m_vbo = vbo;
    geom.m_nIndices = nIndices;
    geom.m_ibo = ibo;
    geom.m_monoColourShader = monoColorShader;
    geom.m_baseColour = baseColour;

    glGenVertexArrays(1, &geom.m_vao);
    glBindVertexArray(geom.m_vao);

    // 1rst attribute buffer : vertices
    glBindBuffer(GL_ARRAY_BUFFER, geom.m_vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,        // attribute
                          3,        // size
                          GL_FLOAT, // type
                          GL_FALSE, // normalized?
                          0,        // stride
                          (void *)0 // array buffer offset
    );
    glBindVertexArray(0);
}

void drawGeometryViewer(GeometryViewer const &geom,
                        Eigen::Matrix4f &cameraMat)
{
    // Set up shader
    glUseProgram(*geom.m_monoColourShader);

    // Set uniforms
    GLuint mvpID = glGetUniformLocation(*geom.m_monoColourShader, "MVP");
    glUniformMatrix4fv(mvpID, 1, GL_FALSE, cameraMat.data());

    GLuint baseColID =
        glGetUniformLocation(*geom.m_monoColourShader, "base_colour");
    glUniform3fv(baseColID, 1, geom.m_baseColour.data());

    glBindVertexArray(geom.m_vao);

    // Index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, geom.m_ibo);

    glDrawElements(GL_TRIANGLES,      // mode
                   geom.m_nIndices,   // count
                   GL_UNSIGNED_SHORT, // type
                   (void *)0          // element array buffer offset
    );
}
