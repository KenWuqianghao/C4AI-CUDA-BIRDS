#include <stdio.h>
#include <cuda.h>
__global__ void maxElementKernel(float *vec, float *result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float sdata[];

    // Perform reduction to find the max element
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (threadIdx.x == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

__global__ void sumOfMaxElements(float *vec1, float *vec2, float *result, int n) {
    __shared__ float max1;
    __shared__ float max2;

    // Find max of first vector
    maxElementKernel<<<1, blockDim.x, blockDim.x * sizeof(float)>>>(vec1, &max1, n);
    cudaDeviceSynchronize();

    // Find max of second vector
    maxElementKernel<<<1, blockDim.x, blockDim.x * sizeof(float)>>>(vec2, &max2, n);
    cudaDeviceSynchronize();

    // Sum the max elements and store the result
    if (threadIdx.x == 0) {
        result[0] = max1 + max2;
    }
}