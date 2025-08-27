cmake_minimum_required(VERSION 3.15)
project(opencv_cpp)

find_package(PCL REQUIRED COMPONENTS io visualization)

add_executable(opencv_cpp main.cpp)
target_link_libraries(opencv_cpp PRIVATE ${PCL_LIBRARIES})
target_include_directories(opencv_cpp PRIVATE ${PCL_INCLUDE_DIRS})
