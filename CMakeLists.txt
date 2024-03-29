cmake_minimum_required(VERSION 2.8.3)
project(cslibs_image_align)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(cmake/cslibs_image_align_enable_c++11.cmake)
include(cmake/cslibs_image_align_extras.cmake)
include(cmake/cslibs_image_align_show_headers.cmake)
include(cmake/cslibs_image_align_add_unit_test_gtest.cmake)


find_package(catkin REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
    INCLUDE_DIRS    include
    CFG_EXTRAS      cslibs_image_align_extras.cmake
                    cslibs_image_align_enable_c++11.cmake
                    cslibs_image_align_show_headers.cmake
                    cslibs_image_align_add_unit_test_gtest.cmake)

include_directories(
   include
   ${OpenCV_INCLUDE_DIRS}
   ${catkin_INCLUDE_DIRS}
)

cslibs_image_align_show_headers()

catkin_add_gtest(${PROJECT_NAME}-test test/test.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)

if(TARGET ${PROJECT_NAME}-test)
  target_link_libraries(${PROJECT_NAME}-test ${OpenCV_LIBRARIES})
endif()


add_executable(example_align
    test/example.cpp
)

target_link_libraries(example_align
    ${OpenCV_LIBRARIES}
    )


add_custom_command(
        TARGET example_align POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/test/img3.png
                ${CMAKE_BINARY_DIR}/devel/lib/${PROJECT_NAME}/img3.png)

add_custom_command(
        TARGET example_align POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/test/img4.png
                ${CMAKE_BINARY_DIR}/devel/lib/${PROJECT_NAME}/img4.png)



install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
