#include <string>
#include <iostream>
#include "Utils/OpenGl.h"
#include "GLShader.h"

using InfoLogGetter=decltype(glGetShaderInfoLog);
using HandleGetter=decltype(glGetShaderiv);

std::string getShaderError(GLuint handle,
    HandleGetter getHandle,
    InfoLogGetter getInfoLog)
{
    GLint maxLength = 0;
    getHandle(handle, GL_INFO_LOG_LENGTH, &maxLength);

    // The maxLength includes the NULL character
    char* infoLog = new char [maxLength];

    getInfoLog(handle, maxLength, &maxLength, infoLog);


    auto infoLogString = std::string(infoLog);
    delete[] infoLog;

    std::cout<<infoLogString<<std::endl;
    return infoLogString;
}

GLuint compileShaderProgram(std::string const & vertexSource,
    std::string const & fragmentSource)
{

    fprintf(stdout, "Compiling vertex shader.\n");

    // Create an empty vertex shader handle
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);

    fprintf(stdout, "Getting shader source.\n");
    const GLchar *vertexSource_c_str = (const GLchar *)vertexSource.c_str();
    // Send the vertex shader source code to GL
    glShaderSource(vertexShader, 1, &vertexSource_c_str, 0);

    fprintf(stdout, "Compiling shader.\n");
    // Compile the vertex shader
    glCompileShader(vertexShader);

    GLint isCompiled = 0;

    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &isCompiled);

    if(isCompiled == GL_FALSE)
    {
        fprintf(stdout, "Compilation failed.\n");
        auto infoLogString = getShaderError(vertexShader, 
            glGetShaderiv, glGetShaderInfoLog);

        // Use the infoLog as you see fit.
        fprintf(stderr, "Error, vertex shader failed to compile:\n%s\n",
           infoLogString.c_str());

        // We don't need the shader anymore.
        glDeleteShader(vertexShader);

        // In this simple program, we'll just leave
        return 0;
    }

    fprintf(stdout, "Compiling fragment shader.\n");
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    
    const GLchar * fragmentSource_c_str = (const GLchar *)fragmentSource.c_str();

    // Send the fragment shader source code to GL
    glShaderSource(fragmentShader, 1, &fragmentSource_c_str, 0);
    
    // Compile the fragment shader
    glCompileShader(fragmentShader);
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &isCompiled);

    if (isCompiled == GL_FALSE)
    {
        auto infoLogString = getShaderError(fragmentShader,
            glGetShaderiv, glGetShaderInfoLog);

        // Use the infoLog as you see fit.
        fprintf(stderr, "Error, fragment shader failed to compile:\n%s\n",
            infoLogString.c_str());

        // We don't need the shader anymore.
        glDeleteShader(fragmentShader);

        // In this simple program, we'll just leave
        return 0;
    }

    fprintf(stdout, "Linking shaders into program.\n");
    // Vertex and fragment shaders are successfully compiled.
    // Now time to link them together into a program.
    // Get a program object.
    GLuint program = glCreateProgram();
    
    // Attach our shaders to our program
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    
    // Link our program
    glLinkProgram(program);
    
    // Note the different functions here: glGetProgram* instead of glGetShader*
    GLint isLinked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, (int *)&isLinked);
    if (isLinked == GL_FALSE)
    {
        auto infoLogString = getShaderError(program,
            glGetProgramiv, glGetProgramInfoLog);

        // We don't need the program anymore.
        glDeleteProgram(program);
        // Don't leak shaders either.
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        // Use the infoLog as you see fit.
        fprintf(stderr, "Error, program failed to link:\n%s\n",
            infoLogString.c_str());

        // In this simple program, we'll just leave
        return 0;
    }
    // Always detach shaders after a successful link.
    glDetachShader(program, vertexShader);
    glDetachShader(program, fragmentShader);

    return program;
}