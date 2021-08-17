#include "cdga/Demo.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

}

}  // namespace cdga

