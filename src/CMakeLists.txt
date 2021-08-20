cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)

project(cdga CXX CUDA)

find_package(CUDAToolkit REQUIRED) # CMake >= 3.17

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

# If DLOPEN is being used for cuda, set a target definintion
target_compile_definitions(${PROJECT_NAME} PRIVATE "USE_DLOPEN_CUDA")

# No need to link against CUDA::cudart or CUDA::cudart_static as we are using the CUDA CMake project language.
# target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)

# Link against the CUDA driver api unless we are dlopening it on linux.
if(NOT USE_DLOPEN_CUDA)
    target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cuda_driver)
endif()


# If useing dlopen, we need to link against it. 
if(USE_DLOPEN_CUDA)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_DL_LIBS})
endif()

# @todo - mostly just for demonstration, but might want to dlopen these later in isolation of the main project. 
# if(TARGET CUDA::nvToolsExt)
#     target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::nvToolsExt)
# endif()
# if(TARGET CUDA::curand)
#     target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::curand)
# endif()
# if(TARGET CUDA::nvrtc)
#     target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::nvrtc)
# endif()