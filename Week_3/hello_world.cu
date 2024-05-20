#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void helloFromGPU()
{
    /*
        We can't use cout in device. Why?
        Because The GPU does not have access to standard
        output streams like std::cout, which are managed
        by the host(CPU) operating system.
    */
    printf("Hello World from GPU!\n");
}

int main()
{
    // Print from host
    cout<<"Hello World from CPU!"<<"\n";

    /*
        Launch a kernel on the GPU with one thread to
        print from GPU
    */
    helloFromGPU<<<1, 1>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    return 0;
}