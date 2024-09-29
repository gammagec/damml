cmake_minimum_required(VERSION 3.28)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED  YES)
set(CMAKE_CXX_EXTENSIONS NO)

project(damml)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

enable_testing()

add_subdirectory(src)
add_subdirectory(examples)