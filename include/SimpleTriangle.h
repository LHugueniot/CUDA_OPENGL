#ifndef SIMPLE_TRIANGLE_H
#define SIMPLE_TRIANGLE_H

#include <vector>

#include "Utils/Eigen.h"
#include "Utils/OpenGL.h"
#include "Utils/General.h"
#include "GLShader.h"


static const char *STVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 MVP;

void main()
{
    //gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0) * MVP;
    //gl_Position = MVP * vec4(aPos.x, aPos.y, aPos.z, 1.0);
    //gl_Position = MVP * vec4(aPos, 1.0);
}

)";
static const char *STFragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
uniform vec4 ourColor;
void main()
{
    //FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
    FragColor = ourColor;
})";

struct SimpleTriangle
{
    SimpleTriangle()
    {
        this->m_vertices =
        {
            -0.5f, -0.5f, 0.0f, // left  
             0.5f, -0.5f, 0.0f, // right 
             0.0f,  0.5f, 0.0f  // top   
        };
	    this->m_shader = compileShaderProgram(
            STVertexShaderSource, STFragmentShaderSource);
        if (this->m_shader == 0)
            throw std::runtime_error("Compiler source did not compile.");
            
        glGenVertexArrays(1, &(this->m_VAO));
        glGenBuffers(1, &(this->m_VBO));
        // bind the Vertex Array Object first, then bind and set vertex
        // buffer(s), and then configure vertex attributes(s).
        glBindVertexArray((this->m_VAO));
        updateVBO();

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
            3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        // note that this is allowed, the call to glVertexAttribPointer
        // registered VBO as the vertex attribute's bound vertex buffer
        // object so afterwards we can safely unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0); 

        // You can unbind the VAO afterwards so other VAO calls won't
        // accidentally modify this VAO), but this rarely happens.
        // Modifying other VAOs requires a call to glBindVertexArray
        // anyways so we generally don't unbind VAOs (nor VBOs) when
        // it's not directly necessary.
        glBindVertexArray(0); 
    }

    void updateVBO()
    {
        glBindBuffer(GL_ARRAY_BUFFER, (this->m_VBO));
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * this->m_vertices.size(),
            this->m_vertices.data(), GL_STATIC_DRAW);
    }

    void draw(ei::Matrix4f const & VP)//, double time)
    {
        glUseProgram(this->m_shader);
        GLuint mvpID = glGetUniformLocation(this->m_shader, "MVP");
        //GLuint testID = glGetUniformLocation(this->m_shader, "test");
        //GLuint MVP2ID = glGetUniformLocation(this->m_shader, "MVP2");
        //std::cout<<"mvpID: "<<mvpID<<std::endl;
        //std::cout<<"testID: "<<testID<<std::endl;
        //std::cout<<"MVP2ID: "<<MVP2ID<<std::endl;
        //std::cout<<"VP.data(): "<<std::endl;
        //for (int i = 0 ; i<16 ; i++)
        //    std::cout<<VP.data()[i]<<", ";
        //std::cout<<std::endl;
            
        
        glUniformMatrix4fv(mvpID, 1, GL_FALSE, VP.data());

        GLuint colID = glGetUniformLocation(
            this->m_shader, "ourColor");
        //std::cout<<"colID: "<<colID<<std::endl;
        //float greenValue = static_cast<float>(sin(timeValue) / 2.0 + 0.5);
        float greenValue = 1.0;
        glUniform4f(colID, 0.0f, greenValue, 0.0f, 1.0f);
//
        // draw our first triangle
        glBindVertexArray(this->m_VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    std::vector<float> m_vertices;
    GLuint m_shader;
    GLuint m_VBO, m_VAO;
};

#endif /* SIMPLE_TRIANGLE_H */