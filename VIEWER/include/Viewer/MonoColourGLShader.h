#ifndef MONO_COLOUR_GL_SHADER_H
#define MONO_COLOUR_GL_SHADER_H

#include "Utils/OpenGl.h"
#include "Viewer/GLShader.h"

static const char *vertexMonoColourSource = R"V0G0N(
#version 330
layout(location = 0) in vec3 vertex_position;
uniform vec3 base_colour;
uniform mat4 MVP;
out vec3 frag_colour;
void main() {
    frag_colour = vertex_position;
    gl_Position = MVP * vec4(vertex_position, 1.0);
};
)V0G0N";

static const char *fragmentMonoColourSource = R"V0G0N(
#version 330
in vec3 frag_colour;
out vec4 colour;
void main() {
  colour = vec4(frag_colour, 1.0);
};
)V0G0N";

GLuint compileMonoColourShaderProgram();

#endif /* MONO_COLOUR_GL_SHADER_H */