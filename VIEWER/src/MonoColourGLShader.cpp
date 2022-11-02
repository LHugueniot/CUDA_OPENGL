#include "Utils/OpenGl.h"
#include "Viewer/MonoColourGLShader.h"

GLuint compileMonoColourShaderProgram(){
    return compileShaderProgram(vertexMonoColourSource, fragmentMonoColourSource);
}