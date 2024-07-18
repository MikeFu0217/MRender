#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * Host main routine
 */
int test_cuda(void)
{

    int dev_num = 0;
    cudaError_t error_id = cudaGetDeviceCount(&dev_num);

    // This function call returns 0 if there are no CUDA capable devices.
    if (dev_num == 0)
    {
        return 0;
    }
    return 1;
}
