include(CTest)
enable_testing()

set(TESTS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TESTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Create test executable
add_executable(cuda_opengl_tests_main ${TESTS_SRC}/main.cu)

target_compile_options(
    cuda_opengl_tests_main

    PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    >)

set_target_properties(cuda_opengl_tests_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Asset setup
target_compile_definitions(cuda_opengl_tests_main PUBLIC ASSETS_DIRECTORY=${ASSETS_DIRECTORY})

# Link internal module libs
target_link_libraries(
    cuda_opengl_tests_main

    PUBLIC
    Utils
    Geometry
    Viewer
    Sim
)

# GLM setup
find_package(glm CONFIG REQUIRED)

# Gtest setup
find_package(GTest CONFIG REQUIRED)

# Link external module libs
target_link_libraries(
    cuda_opengl_tests_main
    PRIVATE
    GTest::gtest
    GTest::gtest_main
    glm::glm
)
add_test(
    NAME cuda_opengl_tests
    COMMAND cuda_opengl_tests_main.exe)

# include(GoogleTest)
# gtest_discover_tests(cuda_opengl_tests_main)
