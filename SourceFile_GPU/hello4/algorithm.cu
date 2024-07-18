#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


__global__ void
hello()
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    printf("hello from GPU block %d/%d, thread %d/%d\n", blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}


/**
 * Host main routine
 */
int test_cuda(void)
{

    hello<<<2, 3>>>();

    cudaDeviceSynchronize();

    printf("Done\n");

    return 0;
}
