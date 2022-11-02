set(SIMULATION_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SIMULATION_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(
    Sim

    STATIC
    ${SIMULATION_SRC}/PBD/PBDSolver.cu
)

target_include_directories(
    Sim

    PUBLIC
    ${SIMULATION_INCLUDE}
)

target_link_libraries(
    Sim

    PUBLIC
    Geometry
)
