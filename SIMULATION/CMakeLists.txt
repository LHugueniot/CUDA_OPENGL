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
target_compile_definitions(Sim
    PRIVATE

    # If the debug configuration pass the DEBUG define to the compiler
    $<$<CONFIG:Debug>:-DDEBUG>>
)
target_link_libraries(
    Sim

    PUBLIC
    Geometry
)
