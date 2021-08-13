cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)

project(cdga CXX CUDA)

get_filename_component(ROOT ${CMAKE_CURRENT_LIST_DIR}/.. REALPATH)

set(SRC_INCLUDE
    ${ROOT}/include/cdga/version.h
    ${ROOT}/include/cdga/Demo.h
    ${ROOT}/include/cdga/cdga.h
)

set(SRC
    ${ROOT}/src/cdga/version.cpp
    ${ROOT}/src/cdga/Demo.cu
)

set(ALL_SRC
    ${SRC_INCLUDE}
    ${SRC}
)

# Create the library target
add_library(${PROJECT_NAME} STATIC ${ALL_SRC})

# Enable PIC
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME}  PUBLIC  "${ROOT}/include")
target_include_directories(${PROJECT_NAME}  PRIVATE "${ROOT}/src")

target_link_libraries(${PROJECT_NAME} PUBLIC cuda)