cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)
include(CMakeDependentOption)

project(cmake-docker-github-actions LANGUAGES CXX CUDA)

# Set the defualt CMAKE BUILD TYPE to Release (for make-like generators)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set the binary output location
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)

# Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "Control the output of compile_commands.json")


# CMake Options.
option(BUILD_TESTS "Enable building tests" OFF)
option(BUILD_SWIG_PYTHON "Enable python bindings via SWIG" OFF)
cmake_dependent_option(BUILD_SWIG_PYTHON_VENV "Enable the use of a venv for swig/python installation" ON "BUILD_SWIG_PYTHON" ON)


add_subdirectory(src "${PROJECT_BINARY_DIR}/cdga")

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_SWIG_PYTHON)
    add_subdirectory(swig)
endif()
