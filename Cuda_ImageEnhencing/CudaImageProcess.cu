#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include <iostream>
#include <cstdint>

#include "typedef.h"
#include "BmpReader.h"
#include "CudaImageProcess.h"



constexpr int BLOCK = 16;
constexpr int BLUR_RADIUS = 1;
constexpr int BLUR_INTENSITY = 6;
constexpr int ENHENCE_STRENGTH = 1;
constexpr int SHARED_MEM_SIZE = 16;
constexpr int KAWASE_PASS_COUNT = 1;


inline bool CudaCheck(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error : " << cudaGetErrorString(err) << "at line " << __LINE__ << "\n";
        return false;
    }
    return true;
}

inline bool CudaKernelCheck(void)
{
    return CudaCheck(cudaGetLastError());
}



__global__ void bilinearReduce(uint8_t* dst, uint8_t* src, size_t pitch, uint32_t width, uint32_t height);
__global__ void bilinearIncrease(uint8_t* dst, uint8_t* src, size_t pitch, uint32_t width, uint32_t height);
__global__ void kawaseBlur(uint8_t* dst, uint8_t* src, size_t pitch, int offset, uint32_t width, uint32_t height);
__global__ void unsharp(int strength , uint8_t* dst, uint8_t* src, size_t pitch, uint8_t* origin, uint32_t width, uint32_t height);


// TODO : (IMP) complete cuda kernel function.


CudaImageProcess::CudaImageProcess(void)
{
    h_rBuf = nullptr;
    h_gBuf = nullptr;
    h_bBuf = nullptr;
    width = height = 0;
    
    d_rBuf = nullptr;
    d_gBuf = nullptr;
    d_bBuf = nullptr;

    d_r2 = nullptr;
    d_g2 = nullptr;
    d_b2 = nullptr;

    d_originalR = nullptr;
    d_originalG = nullptr;
    d_originalB = nullptr;

    pitch = 0;
}

CudaImageProcess::~CudaImageProcess(void)
{
    Close();
}



bool CudaImageProcess::Initialize(void)
{
    bool result = false;

        if (!Bmp::LoadRGBs(&h_rBuf, &h_gBuf, &h_bBuf, static_cast<unsigned int>(width), static_cast<unsigned int>(height)))
        {
            goto LB_RETURN;
        }

        if (!CudaCheck(cudaMallocPitch(&d_rBuf, &pitch, static_cast<size_t>(width), static_cast<size_t>(height))))
        {
            goto LB_FREEALLOC;
        }
        if (!CudaCheck(cudaMallocPitch(&d_gBuf, &pitch, static_cast<size_t>(width), static_cast<size_t>(height))))
        {
            goto LB_FREEALLOC;
        }
        if (!CudaCheck(cudaMallocPitch(&d_bBuf, &pitch, static_cast<size_t>(width), static_cast<size_t>(height))))
        {
            goto LB_FREEALLOC;
        }


        if (!CudaCheck(cudaMallocPitch(&d_r2, &pitch, static_cast<size_t>(width), static_cast<size_t>(height))))
        {
            goto LB_FREEALLOC;
        }
        if (!CudaCheck(cudaMallocPitch(&d_g2, &pitch, static_cast<size_t>(width), static_cast<size_t>(height))))
        {
            goto LB_FREEALLOC;
        }
        if (!CudaCheck(cudaMallocPitch(&d_b2, &pitch, static_cast<size_t>(width), static_cast<size_t>(height))))
        {
            goto LB_FREEALLOC;
        }


        if (!CudaCheck(cudaMallocPitch(&d_originalR, &pitch, static_cast<size_t>(width), static_cast<size_t>(height))))
        {
            goto LB_FREEALLOC;
        }
        if (!CudaCheck(cudaMallocPitch(&d_originalG, &pitch, static_cast<size_t>(width), static_cast<size_t>(height))))
        {
            goto LB_FREEALLOC;
        }
        if (!CudaCheck(cudaMallocPitch(&d_originalB, &pitch, static_cast<size_t>(width), static_cast<size_t>(height))))
        {
            goto LB_FREEALLOC;
        }


        if (!CudaCheck(cudaMemcpy2D(d_rBuf, pitch, h_rBuf->data, sizeof(uint8_t) * width, sizeof(uint8_t) * width, height, cudaMemcpyHostToDevice)))
        {
            goto LB_FREEALLOC;
        }
        if (!CudaCheck(cudaMemcpy2D(d_gBuf, pitch, h_gBuf->data, sizeof(uint8_t) * width, sizeof(uint8_t) * width, height, cudaMemcpyHostToDevice)))
        {
            goto LB_FREEALLOC;
        }
        if (!CudaCheck(cudaMemcpy2D(d_bBuf, pitch, h_bBuf->data, sizeof(uint8_t) * width, sizeof(uint8_t) * width, height, cudaMemcpyHostToDevice)))
        {
            goto LB_FREEALLOC;
        }


        if (!CudaCheck(cudaMemcpy2D(d_originalR, pitch, h_rBuf->data, sizeof(uint8_t) * width, sizeof(uint8_t) * width, height, cudaMemcpyHostToDevice)))
        {
            goto LB_FREEALLOC;
        }
        if (!CudaCheck(cudaMemcpy2D(d_originalG, pitch, h_gBuf->data, sizeof(uint8_t) * width, sizeof(uint8_t) * width, height, cudaMemcpyHostToDevice)))
        {
            goto LB_FREEALLOC;
        }
        if (!CudaCheck(cudaMemcpy2D(d_originalB, pitch, h_bBuf->data, sizeof(uint8_t) * width, sizeof(uint8_t) * width, height, cudaMemcpyHostToDevice)))
        {
            goto LB_FREEALLOC;
        }

result = true;
goto LB_RETURN;

LB_FREEALLOC:

const char* errName = cudaGetErrorName(cudaGetLastError());
const char* errCode = cudaGetErrorString(cudaGetLastError());
fprintf(stderr, "initialize failed with error : %s(%s)\n", errName, errCode);

if (d_rBuf) { cudaFree(d_rBuf);}
if (d_gBuf) { cudaFree(d_gBuf);}
if (d_bBuf) { cudaFree(d_bBuf);}
if (d_originalR) { cudaFree(d_originalR); }
if (d_originalG) { cudaFree(d_originalG); }
if (d_originalB) { cudaFree(d_originalB); }


LB_RETURN:

    return result;
}

bool CudaImageProcess::ImageBlur(void)
{
    if (!applyKawase(KAWASE_PASS_COUNT))
    {
        const char* errname = cudaGetErrorName(cudaGetLastError());
        const char* errcode = cudaGetErrorString(cudaGetLastError());
        fprintf(stderr, "applykawase failed with error : %s(%s)\n", errname, errcode);

        return false;
    }

    return true;
}

bool CudaImageProcess::IncreaseResolution(void)
{
    dim3 block(BLOCK, BLOCK);
    dim3 grid((width + BLOCK - 1) / BLOCK, (width + BLOCK - 1) / BLOCK);
   
    unsharp <<<grid, block>>>(ENHENCE_STRENGTH, d_r2, d_rBuf, pitch, d_originalR, width, height);
    unsharp <<<grid, block>>>(ENHENCE_STRENGTH, d_g2, d_gBuf, pitch, d_originalG, width, height);
    unsharp <<<grid, block>>>(ENHENCE_STRENGTH, d_b2, d_bBuf, pitch, d_originalB, width, height);

    if (!CudaKernelCheck())
    {
        const char* errName = cudaGetErrorName(cudaGetLastError());
        const char* errCode = cudaGetErrorString(cudaGetLastError());
        fprintf(stderr, "unsharp failed with error : %s(%s)\n", errName, errCode);
    }

    return true;
}

bool CudaImageProcess::StoreImage(void)
{
    bool result = false;

    if (!CudaCheck(cudaMemcpy2D(h_rBuf->data, width, d_r2, pitch, width, height, cudaMemcpyDeviceToHost)))
    {
        goto LB_RETURN;
    }
    if (!CudaCheck(cudaMemcpy2D(h_gBuf->data, width, d_g2, pitch, width, height, cudaMemcpyDeviceToHost)))
    {
        goto LB_RETURN;
    }
    if (!CudaCheck(cudaMemcpy2D(h_bBuf->data, width, d_b2, pitch, width, height, cudaMemcpyDeviceToHost)))
    {
        goto LB_RETURN;
    }
    
    if (!Bmp::StoreRGBs("unsharpedLena.bmp", width, height, &h_rBuf, &h_gBuf, &h_bBuf))
    {
        goto LB_RETURN;
    }

    
        result = true;


LB_RETURN:
    return result;
}

void CudaImageProcess::Close(void)
{
    if (d_rBuf) { cudaFree(d_rBuf); }
    if (d_gBuf) { cudaFree(d_gBuf); }
    if (d_bBuf) { cudaFree(d_bBuf); }

    if (d_originalR) { cudaFree(d_originalR); }
    if (d_originalG) { cudaFree(d_originalG); }
    if (d_originalB) { cudaFree(d_originalB); }
}

bool CudaImageProcess::BilinearIncrease(void)
{
    bool result = false;

    dim3 block(BLOCK, BLOCK); // this means threads per block ( 256 threads per block )

    for (int i = 0; i < KAWASE_PASS_COUNT; ++i)
    {
        width *= 2;
        height *= 2;

        dim3 grid((width + BLOCK - 1) / BLOCK, (width + BLOCK - 1) / BLOCK);
        bilinearIncrease << <grid, block >> > (d_r2, d_rBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearIncrease << <grid, block >> > (d_g2, d_gBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearIncrease << <grid, block >> > (d_b2, d_bBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }

        width *= 2;
        height *= 2;

        dim3 grid2((width + BLOCK - 1) / BLOCK, (width + BLOCK - 1) / BLOCK);
        bilinearIncrease << <grid2, block >> > (d_rBuf, d_r2, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearIncrease << <grid2, block >> > (d_gBuf, d_g2, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearIncrease << <grid2, block >> > (d_bBuf, d_b2, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
    }

    result = true;

LB_RETURN:

    return result;
}

bool CudaImageProcess::BilinearReduce(void)
{
    bool result = false;

    dim3 block(BLOCK, BLOCK); // this means threads per block ( 256 threads per block )

    for (int i = 0; i < KAWASE_PASS_COUNT; ++i)
    {
        dim3 reduceGrid((width + BLOCK - 1) / BLOCK, (width + BLOCK - 1) / BLOCK);
        bilinearReduce << <reduceGrid, block >> > (d_r2, d_rBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearReduce << <reduceGrid, block >> > (d_g2, d_gBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearReduce << <reduceGrid, block >> > (d_b2, d_bBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }

        width /= 2;
        height /= 2;

        dim3 reduceGrid2((width  + BLOCK - 1) / BLOCK, (width + BLOCK - 1) / BLOCK);
        bilinearReduce << <reduceGrid2, block >> > (d_rBuf, d_r2, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearReduce << <reduceGrid2, block >> > (d_gBuf, d_g2, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearReduce << <reduceGrid2, block >> > (d_bBuf, d_b2, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }

        width /= 2;
        height /= 2;
    }

    result = true;

LB_RETURN:

    return result;
}





bool CudaImageProcess::applyKawase(uint32_t applyTimes)
{
    bool result = false;

    int offset = 1;
    dim3 block(BLOCK, BLOCK); // this means threads per block ( 256 threads per block )


    for (uint32_t i = 0; i < applyTimes; ++i)
    {
        dim3 reduceGrid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
        bilinearReduce <<<reduceGrid, block >>> (d_r2, d_rBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearReduce <<<reduceGrid, block >>> (d_g2, d_gBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearReduce <<<reduceGrid, block >>> (d_b2, d_bBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }

        width /= 2;
        height /= 2;

        dim3 kawaseGrid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
        kawaseBlur <<<kawaseGrid, block>>> (d_rBuf, d_r2, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        kawaseBlur <<<kawaseGrid, block>>> (d_gBuf, d_g2, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        kawaseBlur <<<kawaseGrid, block>>> (d_bBuf, d_b2, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        
        offset++;

        dim3 reduceGrid2((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
        bilinearReduce << <reduceGrid2, block >> > (d_r2, d_rBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearReduce << <reduceGrid2, block >> > (d_g2, d_gBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearReduce << <reduceGrid2, block >> > (d_b2, d_bBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }

        width /= 2;
        height /= 2;

        dim3 kawaseGrid2((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
        kawaseBlur << <kawaseGrid2, block >> > (d_rBuf, d_r2, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        kawaseBlur << <kawaseGrid2, block >> > (d_gBuf, d_g2, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        kawaseBlur << <kawaseGrid2, block >> > (d_bBuf, d_b2, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }

        ++offset;
    }

    for (uint32_t i = 0; i < applyTimes; ++i)
    { 
        width *= 2;
        height *= 2;

        dim3 increaseGrid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
        bilinearIncrease << <increaseGrid, block >> > (d_r2, d_rBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearIncrease << <increaseGrid, block >> > (d_g2, d_gBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearIncrease << <increaseGrid, block >> > (d_b2, d_bBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }

        dim3 kawaseGrid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
        kawaseBlur << <kawaseGrid, block >> > (d_rBuf, d_r2, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        kawaseBlur << <kawaseGrid, block >> > (d_gBuf, d_g2, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        kawaseBlur << <kawaseGrid, block >> > (d_bBuf, d_b2, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }

        --offset;

        width *= 2;
        height *= 2;
        
        dim3 increaseGrid2 ((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
        bilinearIncrease << <increaseGrid2, block >> > (d_r2, d_rBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearIncrease << <increaseGrid2, block >> > (d_g2, d_gBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearIncrease << <increaseGrid2, block >> > (d_b2, d_bBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }

        dim3 kawaseGrid2((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
        kawaseBlur << <kawaseGrid2, block >> > (d_rBuf, d_r2, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        kawaseBlur << <kawaseGrid2, block >> > (d_gBuf, d_g2, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        kawaseBlur << <kawaseGrid2, block >> > (d_bBuf, d_b2, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }

        --offset;
    }

    result = true;

LB_RETURN:

    return result;
}


// blockDim.x, blockDim.y, blockDim.z: dimensions of threads in a block
// gridDim.x, gridDim.y, gridDim.z   : dimensions of blocks in the grid
// blockIdx.x, blockIdx.y, blockIdx.z: block index in each dimension
// threadIdx.x, threadIdx.y, threadIdx.z: thread index in each dimension


__global__ void bilinearReduce(uint8_t* dst, uint8_t* src, size_t pitch, uint32_t width, uint32_t height)
{
    const uint32_t bx = blockDim.x * blockIdx.x;
    const uint32_t by = blockDim.y * blockIdx.y;
    const uint32_t outX= bx + threadIdx.x;
    const uint32_t outY = by + threadIdx.y;

    const uint32_t reduceW = width >> 1;
    const uint32_t reduceH = height >> 1;

    if (outX >= reduceW || outY >= reduceH || outX < 0 || outY < 0) return;

    const uint32_t srcX = outX << 1;
    const uint32_t srcY = outY << 1;

    uint8_t tl = src[srcX + pitch * srcY];
    uint8_t tr = src[srcX + 1 + pitch * srcY];
    uint8_t bl = src[srcX + pitch * (srcY + 1)];
    uint8_t br = src[srcX + 1 + pitch * (srcY + 1)];

    uint32_t sum = tl + tr + bl + br;
    uint8_t calV = sum >> 2;

    dst[outX + pitch * outY] = calV;
}

__global__ void bilinearIncrease(uint8_t* dst, uint8_t* src, size_t pitch, uint32_t width, uint32_t height)
{
    const uint32_t bx = blockDim.x * blockIdx.x;
    const uint32_t by = blockDim.y * blockIdx.y;
    const uint32_t outX = bx + threadIdx.x;
    const uint32_t outY = by + threadIdx.y;

    const uint32_t increaseW = width << 1;
    const uint32_t increaseH = height << 1;

    if (outX >= increaseW || outY >= increaseH || outX < 0 || outY < 0) return;

    float fx = outX * 0.5f;
    float fy = outY * 0.5f;

    int ix = (int)fx;
    int iy = (int)fy;
    int ix2 = min(ix + 1, width - 1);
    int iy2 = min(iy + 1, height - 1);

    float dx = fx - ix;
    float dy = fy - iy;

    uint8_t tl = src[ix + pitch * iy];
    uint8_t tr = src[ix2 + pitch * iy];
    uint8_t bl = src[ix + pitch * iy2];
    uint8_t br = src[ix2+ pitch * iy2];

    float sum = ((tl * (1 - dx) + tr * dx) * (1 - dy) + (bl * (1 - dx) + br * dx) * dy) + 0.5f;
    uint8_t calV = (uint8_t)(sum);

    dst[outX + pitch * outY] = calV;
}

__global__ void kawaseBlur(uint8_t* dst, uint8_t* src, size_t pitch, int offset, uint32_t width, uint32_t height)
{
    const uint32_t bx = blockIdx.x * blockDim.x;
    const uint32_t by = blockIdx.y * blockDim.y;
    const uint32_t srcX = bx + threadIdx.x;
    const uint32_t srcY = by + threadIdx.y;

    if (srcX >= width || srcY >= height || srcX < 0 || srcY < 0) return;

    int pox = min(srcX + offset, width - 1); // plus offset
    int poy = min(srcY + offset, height - 1);
    int mox = max(srcX - offset, 0); // minus offset
    int moy = max(srcY - offset, 0);

    uint32_t sum = src[pox + pitch * poy] + src[pox + pitch * moy] + src[mox+ pitch * poy] + src[mox + pitch * moy];
    uint32_t avg = sum >> 2;

    dst[srcX + pitch * srcY] = (uint8_t)avg;
}

__global__ void unsharp(int strength, uint8_t* dst, uint8_t* src, size_t pitch, uint8_t* origin, uint32_t width, uint32_t height)
{
    const uint32_t srcX = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t srcY = blockIdx.y * blockDim.y + threadIdx.y;

    if (srcX >= width || srcY >= height || srcX < 0 || srcY < 0) return;

    const uint32_t idx = srcX + pitch * srcY;

    int oColor = origin[idx];
    int bColor = src[idx];

    int diff = oColor - bColor;
    int sharped = oColor + (strength * diff);
    uint8_t calV = max(min(sharped, 255), 0);

    dst[idx] = calV;
}