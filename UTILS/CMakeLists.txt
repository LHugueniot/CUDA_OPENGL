set(UTILS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(UTILS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(
    Utils

    STATIC
    ${UTILS_SRC}/Eigen.cpp
    ${UTILS_SRC}/Mesh.cpp
    ${UTILS_SRC}/OpenGL.cpp
)

target_include_directories(
    Utils

    PUBLIC
    ${UTILS_INCLUDE}
)

target_link_libraries(
    Utils

    PUBLIC
    OpenGL::GL
    glfw
    Eigen3::Eigen
    assimp::assimp
    imgui::imgui
    GLEW::glew
)