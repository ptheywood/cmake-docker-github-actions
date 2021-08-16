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

# Set the C++/CUDA standard universally.
set(CXXSTD 17)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0.0)
    set(CXXSTD 14)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.29)
        # 17 required.
        if(CXXSTD LESS 17)
            message(FATAL_ERROR "MSVC >= 19.29 requires CMake >= 3.18 and CUDA >= 11.0")
        endif()
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.11)
        # 17 available?
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.10)
        # use 14
        set(CXXSTD 14)
    endif()
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD ${CXXSTD})
    set(CMAKE_CXX_STANDARD_REQUIRED true)
endif()
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD ${CXXSTD})
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()


add_subdirectory(src "${PROJECT_BINARY_DIR}/cdga")

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_SWIG_PYTHON)
    add_subdirectory(swig)
endif()
