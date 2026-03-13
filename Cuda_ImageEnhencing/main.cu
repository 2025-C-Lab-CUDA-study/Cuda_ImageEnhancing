#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include <iostream>
#include <cstdint>

#include "CudaImageProcess.h"

inline void CudaCheck(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error : " << cudaGetErrorString(err) << "at line " << __LINE__ <<"\n";
        exit(EXIT_FAILURE);
    }
}

inline void CudaKernelCheck(void)
{
    CudaCheck(cudaGetLastError());
}

int main(void)
{
    CudaImageProcess imageProcessor;

    if (!imageProcessor.Initialize())
    {
        fprintf(stderr, "imageProcessor initialize failed\n");
        return 1;
    }

    if (!imageProcessor.BilinearReduce())
    {
        fprintf(stderr, "imageProcessor reduce failed\n");
        return 1;
    }

    /*if (!imageProcessor.ImageBlur())
    {
        fprintf(stderr, "imageProcessor imageBlur failed\n");
        imageProcessor.Close();
        return 1;
    }
    if (!imageProcessor.IncreaseResolution())
    {
        fprintf(stderr, "imageProcessor IncreaseResolution failed\n");
        imageProcessor.Close();
        return 1;
    }*/
    if (!imageProcessor.StoreImage())
    {
        fprintf(stderr, "imageProcessor StoreImage failed\n");
        imageProcessor.Close();
        return 1;
    }
    imageProcessor.Close();    

    return 0;
}