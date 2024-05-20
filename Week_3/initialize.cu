#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel function to initialize each element of the array with its index
__global__ void initialize_array(int *array)
{
    // Calculate the index for the current thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize the array element at the calculated index with its index value
    array[index] = index;
}

int main()
{
    const int array_size = 10;
    int *d_array;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_array, array_size * sizeof(int));

    // Launch the CUDA kernel to initialize the array
    int threadsPerBlock = 5; // Example: 5 threads per block
    int blocksPerGrid = (array_size + threadsPerBlock - 1) / threadsPerBlock;
    initialize_array<<<blocksPerGrid, threadsPerBlock>>>(d_array);

    // Copy data from device to host
    int h_array[array_size];
    cudaMemcpy(h_array, d_array, array_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the initialized array
    cout << "Initialized Array:" << endl;
    for (int i = 0; i < array_size; ++i) {
        cout << h_array[i] << " ";
    }
    cout << endl;

    // Free GPU memory
    cudaFree(d_array);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    return 0;
}