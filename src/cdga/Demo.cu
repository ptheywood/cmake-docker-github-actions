#include "cdga/Demo.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Include the class which should be in detail, which interacts with dlopen.
// @todo - might need to make this include conditional?
// @todo - should be in detail
#include "cdga/detail/DSOStuff.h"

namespace cdga {



Demo::Demo() : count(0) { }

Demo::~Demo() { }

__global__ void demoKernel(unsigned int count) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int total = blockDim.x * gridDim.x;
    if(idx == 0) {
        printf("Thread %u of %u: count = %u\n", idx, total, count);
    }
}


void Demo::demo() {


    // dlopen testing. This would want separating / doing much nicer.
    #if defined(USE_DLOPEN_CUDA)

    void * libcuda_handle = detail::DSOStuff::OpenLibraryHandle("libcuda.so.1");
    if (libcuda_handle == nullptr) {
        fprintf(stderr, "Bad stuff happened. @todo \n");
        exit(EXIT_FAILURE);
    }
    // Load required libcuda.so methods. If any fail to load, this will exit, so no need to check the result with the current implementation
    
    CUresult (*cuCtxGetCurrent)( CUcontext* );
    CUresult (*cuGetErrorString)( CUresult, const char** );

    cuCtxGetCurrent = (CUresult (*)( CUcontext* )) detail::DSOStuff::SymbolFromLibrary(libcuda_handle, "cuCtxGetCurrent");
    cuGetErrorString = (CUresult (*)( CUresult, const char** )) detail::DSOStuff::SymbolFromLibrary(libcuda_handle, "cuGetErrorString");
    #endif
    this->count++;

    // Initialise a runtime cuda context on the default device.
    cudaFree(0);

    // Get the current cuda context using the driver api, just to make use of the driver api for testing reasons.

    CUresult driverAPIStatus;
    CUcontext ctx;
    driverAPIStatus = cuCtxGetCurrent(&ctx);
    if (driverAPIStatus != CUDA_SUCCESS) {
        const char * errstr;
        cuGetErrorString(driverAPIStatus, &errstr);
        fprintf(stderr, "Error: Cuda driver Error %s at %s::%d\n", errstr, __FILE__, __LINE__);
    } else {
        printf("cuCtxGetCurrent success\n");
    }
    
    demoKernel<<<1, 1>>>(this->count);
    cudaError_t status;
    cudaDeviceSynchronize();
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        fprintf(stderr, "Error: Cuda Error %s at %s::%d\n", cudaGetErrorString(status), __FILE__, __LINE__);
    }

    // @todo move this and do it nicer. 
    #if defined(USE_DLOPEN_CUDA)
        // Close the handle.
        if(libcuda_handle) {
            detail::DSOStuff::CloseLibraryHandle(&libcuda_handle);
        }
    #endif

}

}  // namespace cdga

