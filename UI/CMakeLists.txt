set(UI_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(UI_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(
    Ui

    STATIC
    ${UI_SRC}/GLFWState.cpp
)

target_include_directories(
    Ui

    PUBLIC
    ${UI_INCLUDE}
)

target_link_libraries(
    Ui

    PUBLIC
    Utils
    glfw
)
