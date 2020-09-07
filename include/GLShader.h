#ifndef GL_SHADER_H
#define GL_SHADER_H

#include "Utils.cuh"

GLuint compileShaderProgram(std::string const & vertexSource, std::string const & fragmentSource);

#endif /* GL_SHADER_H */