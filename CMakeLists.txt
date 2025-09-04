cmake_minimum_required(VERSION 3.15)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Your include and src
include_directories(include)
file(GLOB SRC_FILES src/*.cpp main.cpp)
# file(GLOB SRC_FILES test.cpp)

# Find OpenCV (from the build/install folder)
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/libs/opencv/build/install")
find_package(OpenCV REQUIRED)

add_executable(my_project ${SRC_FILES})
target_include_directories(my_project PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(my_project PRIVATE ${OpenCV_LIBS})
