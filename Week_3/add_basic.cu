#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel function to double each element in the array
__global__ void add_basic(int *data, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        data[index] *= 2;
    }
}

int main()
{
    int *h_data;     // Host array
    int *d_data;     // Device array
    int n = 1024;    // Size of the array

    // Allocate host memory
    h_data = (int*)malloc(n * sizeof(int));

    // Initialize host array
    for(int i = 0; i < n; i++) {
        h_data[i] = i;  // Example data
    }

    // Allocate device memory
    cudaMalloc((void**)&d_data, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    add_basic<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy data back from device to host
    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Example output
    for(int i = 0; i < 10; i++) {  // Print the first 10 elements
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_data);

    // Free host memory
    free(h_data);

    return 0;
}