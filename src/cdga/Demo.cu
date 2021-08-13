#include "cdga/Demo.h"

#include <stdio.h>
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
    
    demoKernel<<<1, 1>>>(this->count);
    cudaError_t status;
    cudaDeviceSynchronize();
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        fprintf(stderr, "Error: Cuda Error %s at %s::%d\n", cudaGetErrorString(status), __FILE__, __LINE__);
    }

}

}  // namespace cdga

