#include "cdga/Demo.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// If Using dlopen, include the appropraite header. This is only currently implemented for linux.
#if defined(USE_DLOPEN_CUDA) !defined(_MSC_VER)
    #include <dlfcn.h>
#else 
#include <cuda.h>
#endif // USE_DLOPEN

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

    // Void pointer to store the dlopen handle
    void * libcuda_handle = nullptr;
    // dlopen error codes.
    char * libcuda_error;
    // Stub for method protos. Do this in a separate file to be included?. dysyms are stored in these?
    // Might need to be per version of the .so (i.e. CUDA 11.2 might ahve different protos that need to be stubbed thatn 110)
    // @todo - could these be std::functions?
    CUresult (*cuCtxGetCurrent)( CUcontext* );
    CUresult (*cuGetErrorString)( CUresult, const char** );

    // Open the .so. 
    // @todo - better name calculation.
    libcuda_handle = dlopen("libcuda.so.1", RTLD_LAZY);
    printf("handle opened? %p\n", libcuda_handle);
    if (!libcuda_handle) {
        fprintf(stderr, "dlopen error libcuda_handle: %s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    // Not sure why this is being called again? reset the error incase the handle was oipened perhaps?
    dlerror();
    // Load the symbols into the function pointers.
    cuCtxGetCurrent = (CUresult (*)( CUcontext* )) dlsym(libcuda_handle, "cuCtxGetCurrent");
    libcuda_error = dlerror();
    if (libcuda_error != NULL) {
        fprintf(stderr, "libcuda_error: %s\n", libcuda_error);
        exit(EXIT_FAILURE);
    }
    cuGetErrorString = (CUresult (*)( CUresult, const char** )) dlsym(libcuda_handle, "cuGetErrorString");
    libcuda_error = dlerror();
    if (libcuda_error != NULL) {
        fprintf(stderr, "libcuda_error: %s\n", libcuda_error);
        exit(EXIT_FAILURE);
    }
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
            dlclose(libcuda_handle);
            libcuda_handle = nullptr;
            // @todo - probabyl check for errors again.
        }
    #endif

}

}  // namespace cdga

