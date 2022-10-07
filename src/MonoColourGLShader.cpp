#include "Utils/OpenGl.h"
#include "MonoColourGLShader.h"

GLuint compileMonoColourShaderProgram(){
    return compileShaderProgram(vertexMonoColourSource, fragmentMonoColourSource);
}