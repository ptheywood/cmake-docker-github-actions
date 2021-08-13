cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)

project(tests CXX CUDA)

set(SRC
    ${CMAKE_CURRENT_LIST_DIR}/src/main.cu
)

add_executable(${PROJECT_NAME} ${SRC})

# link against the cdga target, assuming that it exists
target_link_libraries(${PROJECT_NAME} cdga)