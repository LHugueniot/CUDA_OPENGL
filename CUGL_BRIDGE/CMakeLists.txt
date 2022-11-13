set(CUGL_BRIDGE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CUGL_BRIDGE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(
    CuGlBridge

    STATIC
    ${CUGL_BRIDGE_SRC}/Eigen.cpp
    ${CUGL_BRIDGE_SRC}/Mesh.cpp
    ${CUGL_BRIDGE_SRC}/OpenGL.cpp
)

target_include_directories(
    CuGlBridge

    PUBLIC
    ${CUGL_BRIDGE_INCLUDE}
)

target_link_libraries(
    CuGlBridge

    PUBLIC
    Utils
)