add_definitions(-W
                -Wall
                -Wno-unused-parameter
                -Wno-unused-function
                -fno-strict-aliasing
                -Wno-deprecated-register)

message("${PROJECT_NAME} using -march=native!")
add_definitions(-march=native)


if(NOT ${CMAKE_BUILD_TYPE} STREQUAL Debug)
    add_definitions(-Ofast
                     -rdynamic
                     -ffast-math)
    message("[${PROJECT_NAME}]: Compiling with optimization!")
endif()
