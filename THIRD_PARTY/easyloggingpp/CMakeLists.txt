set(EASYLOGGINGPP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(EASYLOGGINGPP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(
    easyloggingpp

    STATIC
    ${EASYLOGGINGPP_SRC}/easylogging++.cc
)

target_include_directories(
    easyloggingpp

    PUBLIC
    ${EASYLOGGINGPP_INCLUDE}
)
