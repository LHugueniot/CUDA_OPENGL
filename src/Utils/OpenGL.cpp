#include <Utils/OpenGL.h>
#include <sstream>

static const std::map<GLenum, std::string> kGLDebugEnumToString {
    {GL_DEBUG_TYPE_ERROR, "GL_DEBUG_TYPE_ERROR"},
    {GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR, "GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR"},
    {GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR, "GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR"},
    {GL_DEBUG_TYPE_PORTABILITY, "GL_DEBUG_TYPE_PORTABILITY"},
    {GL_DEBUG_TYPE_PERFORMANCE, "GL_DEBUG_TYPE_PERFORMANCE"},
    {GL_DEBUG_TYPE_MARKER, "GL_DEBUG_TYPE_MARKER"},
    {GL_DEBUG_TYPE_PUSH_GROUP, "GL_DEBUG_TYPE_PUSH_GROUP"},
    {GL_DEBUG_TYPE_POP_GROUP, "GL_DEBUG_TYPE_POP_GROUP"},
    {GL_DEBUG_TYPE_OTHER, "GL_DEBUG_TYPE_OTHER"}
};

void GLAPIENTRY
GLDebugMessageCallback(GLenum source,
                       GLenum type,
                       GLuint id,
                       GLenum severity,
                       GLsizei length,
                       const GLchar* message,
                       const void* userParam)
{
    std::stringstream errStream;

    errStream<<"GL CALLBACK: "<<(type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR ** " : " ")
             <<kGLDebugEnumToString.at(type)<<" "
             <<severity<<" "
             <<message<<std::endl;

    if (type == GL_DEBUG_TYPE_ERROR)
    {
        throw std::exception(errStream.str().c_str());
    }
    else
    { 
        std::cout<<errStream.str()<<std::endl;
    }
}
