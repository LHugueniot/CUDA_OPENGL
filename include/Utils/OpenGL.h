#ifndef UTIL_OPENGL_H
#define UTIL_OPENGL_H

// Graphics
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <string>

void _checkGLError(const char *file, int line);
 
///
/// Usage
/// [... some opengl calls]
/// glCheckError();
///
#define checkGLError() _checkGLError(__FILE__,__LINE__)
 
inline void _checkGLError(const char *file, int line) {
        GLenum err (glGetError());
 
        while(err!=GL_NO_ERROR) {
                std::string error;
 
                switch(err) {
                        case GL_INVALID_OPERATION:      error="INVALID_OPERATION";      break;
                        case GL_INVALID_ENUM:           error="INVALID_ENUM";           break;
                        case GL_INVALID_VALUE:          error="INVALID_VALUE";          break;
                        case GL_OUT_OF_MEMORY:          error="OUT_OF_MEMORY";          break;
                        case GL_INVALID_FRAMEBUFFER_OPERATION:  error="INVALID_FRAMEBUFFER_OPERATION";  break;
                }
 
                std::cerr << "GL_" << error.c_str() <<" - "<<file<<":"<<line<<std::endl;
                err=glGetError();
        }
}

#endif /* UTIL_OPENGL_H */