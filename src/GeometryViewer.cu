#include "GeometryViewer.cuh"

void initGeometryViewer(GeometryViewer &geom, uint nVertices, GLuint vbo,
                        uint nIndices, GLuint ibo, GLuint *monoColorShader,
                        const ei::Vector3f &baseColour)
{
    assert(vbo > 0);
    assert(ibo > 0);
    geom.m_nVertices = nVertices;
    geom.m_vbo = vbo;
    geom.m_nIndices = nIndices;
    geom.m_ibo = ibo;
    geom.m_monoColourShader = monoColorShader;
    geom.m_baseColour = baseColour;

    glGenVertexArrays(1, &geom.m_vao);
    checkGLError();
    glBindVertexArray(geom.m_vao);
    checkGLError();
    assert(geom.m_vao > 0);

    // 1rst attribute buffer : vertices
    glEnableVertexAttribArray(0);
    checkGLError();
    glBindBuffer(GL_ARRAY_BUFFER, geom.m_vbo);
    checkGLError();
    glVertexAttribPointer(0,        // attribute
                          3,        // size
                          GL_FLOAT, // type
                          GL_FALSE, // normalized?
                          0,        // stride
                          (void *)0 // array buffer offset
    );
    checkGLError();

    // Index buffer
    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, geom.m_ibo);

    // glBindVertexArray(0);
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
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, geom.m_vbo);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, geom.m_ibo);

    assert(geom.m_nIndices > 0);
    glDrawElements(GL_TRIANGLES,      // mode
                   geom.m_nIndices,   // count
                   GL_UNSIGNED_SHORT, // type
                   (void *)0);        // element array buffer offset

    // glBindVertexArray(0);
}
