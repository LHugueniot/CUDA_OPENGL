set(VIEWER_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(VIEWER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(
    Viewer

    STATIC
    ${VIEWER_SRC}/Camera.cpp
    ${VIEWER_SRC}/GeometryViewer.cpp
    ${VIEWER_SRC}/GLShader.cpp
    ${VIEWER_SRC}/MonoColourGLShader.cpp
    ${VIEWER_SRC}/PlaneGLData.cpp
)

target_include_directories(
    Viewer

    PUBLIC
    ${VIEWER_INCLUDE}
)

target_link_libraries(
    Viewer

    PUBLIC
    Utils
)
