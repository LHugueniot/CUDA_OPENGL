set(GEOMETRY_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(GEOMETRY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(
    Geometry

    STATIC
    ${GEOMETRY_SRC}/CuGlBuffer.cu
    ${GEOMETRY_SRC}/CuGlGeometry.cu
    ${GEOMETRY_SRC}/Geometry.cu
    ${GEOMETRY_SRC}/LoadGeometry.cu
)

target_include_directories(
    Geometry

    PUBLIC
    ${GEOMETRY_INCLUDE}
)

target_link_libraries(
    Geometry

    PUBLIC
    Utils
)
