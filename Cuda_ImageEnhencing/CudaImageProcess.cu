#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include <iostream>
#include <cstdint>

#include "typedef.h"
#include "BmpReader.h"
#include "CudaImageProcess.h"



constexpr int BLOCK = 16;
constexpr int BLUR_RADIUS = 1;
constexpr int BLUR_INTENSITY = 150;
constexpr int ENHENCE_STRENGTH = 1;
constexpr int SHARED_MEM_SIZE = 16;
constexpr int KAWASE_PASS_COUNT = 2;


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



__global__ void bilinearReduce(uint8_t* buf, size_t pitch, uint32_t width, uint32_t height);
__global__ void bilinearIncrease(uint8_t* buf, size_t pitch, uint32_t width, uint32_t height);
__global__ void kawaseBlur(uint8_t* buf, size_t pitch, int offset, uint32_t width, uint32_t height);
__global__ void unsharp(int strength , uint8_t* buf, size_t pitch, uint8_t* origin, uint32_t width, uint32_t height);


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
        const char* errName = cudaGetErrorName(cudaGetLastError());
        const char* errCode = cudaGetErrorString(cudaGetLastError());
        fprintf(stderr, "applyKawase failed with error : %s(%s)\n", errName, errCode);

        return false;
    }

    return true;
}

bool CudaImageProcess::IncreaseResolution(void)
{
    dim3 block(BLOCK, BLOCK);
    dim3 grid((width + BLOCK - 1) / BLOCK, (width + BLOCK - 1) / BLOCK);
   
    unsharp <<<grid, block>>>(ENHENCE_STRENGTH, d_rBuf, pitch, d_originalR, width, height);
    unsharp <<<grid, block>>>(ENHENCE_STRENGTH, d_gBuf, pitch, d_originalG, width, height);
    unsharp <<<grid, block>>>(ENHENCE_STRENGTH, d_bBuf, pitch, d_originalB, width, height);

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

    if (!CudaCheck(cudaMemcpy2D(h_rBuf->data, width, d_rBuf, pitch, width, height, cudaMemcpyDeviceToHost)))
    {
        goto LB_RETURN;
    }
    if (!CudaCheck(cudaMemcpy2D(h_gBuf->data, width, d_gBuf, pitch, width, height, cudaMemcpyDeviceToHost)))
    {
        goto LB_RETURN;
    }
    if (!CudaCheck(cudaMemcpy2D(h_bBuf->data, width, d_bBuf, pitch, width, height, cudaMemcpyDeviceToHost)))
    {
        goto LB_RETURN;
    }
    
    if (!Bmp::StoreRGBs("unsharpedLena.bmp", width, height, h_rBuf, h_gBuf, h_bBuf))
    {
        goto LB_RETURN;
    }

    
    result = true;


LB_RETURN:
    return result;
}

void CudaImageProcess::Close(void)
{
    if (h_rBuf) { free(h_rBuf); }
    if (h_gBuf) { free(h_gBuf); }
    if (h_bBuf) { free(h_bBuf); }

    if (d_rBuf) { cudaFree(d_rBuf); }
    if (d_gBuf) { cudaFree(d_gBuf); }
    if (d_bBuf) { cudaFree(d_bBuf); }

    if (d_originalR) { cudaFree(d_originalR); }
    if (d_originalG) { cudaFree(d_originalG); }
    if (d_originalB) { cudaFree(d_originalB); }
}





bool CudaImageProcess::applyKawase(uint32_t applyTimes)
{
    bool result = false;

    int offset = 1;
    dim3 block(BLOCK, BLOCK); // this means threads per block ( 256 threads per block )


    for (uint32_t i = 0; i < applyTimes; ++i)
    {
        dim3 reduceGrid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
        bilinearReduce <<<reduceGrid, block >>> (d_rBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearReduce <<<reduceGrid, block >>> (d_gBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearReduce <<<reduceGrid, block >>> (d_bBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }

        width /= 2;
        height /= 2;

        int kawaseSharedSize = (BLOCK + offset) * (BLOCK + offset);
        dim3 kawaseGrid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
        kawaseBlur <<<kawaseGrid, block, kawaseSharedSize>>> (d_rBuf, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        kawaseBlur <<<kawaseGrid, block, kawaseSharedSize >>> (d_gBuf, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        kawaseBlur <<<kawaseGrid, block, kawaseSharedSize >>> (d_bBuf, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        
        ++offset;
    }

    for (uint32_t i = 0; i < applyTimes; ++i)
    { 
        uint32_t targetW = width << 1;
        uint32_t targetH = height << 1;

        int tempSize = (BLOCK + 2) * (BLOCK + 2);
        dim3 grid((targetW + BLOCK - 1) / BLOCK, (targetW + BLOCK - 1) / BLOCK);
        bilinearIncrease <<<grid, block, tempSize>>> (d_rBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearIncrease <<<grid, block, tempSize >>> (d_gBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        bilinearIncrease <<<grid, block, tempSize >>> (d_bBuf, pitch, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }

        width *= 2;
        height *= 2;

        int kawaseSharedSize = (BLOCK + offset) * (BLOCK + offset);
        dim3 kawaseGrid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
        kawaseBlur << <kawaseGrid, block, kawaseSharedSize>>> (d_rBuf, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        kawaseBlur << <kawaseGrid, block, kawaseSharedSize>>> (d_gBuf, pitch, offset, width, height);
        if (!CudaKernelCheck()) { goto LB_RETURN; }
        kawaseBlur << <kawaseGrid, block, kawaseSharedSize>>> (d_bBuf, pitch, offset, width, height);
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


__global__ void bilinearReduce(uint8_t* buf, size_t pitch, uint32_t width, uint32_t height)
{

    __shared__ uint8_t temp[SHARED_MEM_SIZE][SHARED_MEM_SIZE];

    const int reduceW = width >> 1;
    const int reduceH = height >> 1;

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t bx = blockIdx.x * blockDim.x;
    const uint32_t by = blockIdx.y * blockDim.y;

    uint32_t srcX = bx + tx;
    uint32_t srcY = by + ty;

    uint32_t loadX = max(min(srcX, width - 1), 0);
    uint32_t loadY = max(min(srcY, height - 1), 0);

    temp[tx][ty] = buf[loadX + pitch * loadY];
    
    __syncthreads();

    uint32_t loadtx = max(min(tx, SHARED_MEM_SIZE / 2 - 1), 0);
    uint32_t loadty = max(min(ty, SHARED_MEM_SIZE / 2 - 1), 0);
    uint32_t outX = bx / 2 + loadtx;
    uint32_t outY = by / 2 + loadty;

	if (outX < reduceW && outY < reduceH)
	{
		uint16_t sum = (uint16_t)temp[tx * 2][ty * 2] +
			temp[tx * 2 + 1][ty * 2] +
			temp[tx * 2][ty * 2+ 1] +
			temp[tx * 2+ 1][ty * 2+ 1];
		uint8_t avg = sum >> 2;

		buf[outY * pitch + outX] = avg;
	}
}

__global__ void bilinearIncrease(uint8_t* buf, size_t pitch, uint32_t width, uint32_t height)
{
    extern __shared__ uint8_t temp[];

    const uint32_t increaseW = width << 1;
    const uint32_t increaseH = height << 1;
    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t bx = blockIdx.x * blockDim.x;
    const uint32_t by = blockIdx.y * blockDim.y;
    const uint32_t outX = bx + tx;
    const uint32_t outY = by + ty;

    float srcFx = (float)outX * 0.5f;
    float srcFy = (float)outY * 0.5f;

    float dx = srcFx - (int)srcFx;
    float dy = srcFy - (int)srcFy;
    int srcIX = (int)srcFx;
    int srcIY = (int)srcFy;

    int x = min((int)srcFx + 1, (int)increaseW - 1);
    int y = min((int)srcFy + 1, (int)increaseH - 1);

    uint8_t tl = buf[srcIY * pitch + srcIX];
    uint8_t tr = buf[srcIY * pitch + x];
    uint8_t bl = buf[(srcIY + 1) * pitch + srcIX];
    uint8_t br = buf[(srcIY + 1) * pitch + x];

    float v = (tl * (1 - dx) + tr * dx) * (1 - dy) +
        (bl * (1 - dx) + br * dx) * dy;

    buf[outY * pitch + outX] = (uint8_t)(v + 0.5f);
}

__global__ void kawaseBlur(uint8_t* buf, size_t pitch, int offset, uint32_t width, uint32_t height)
{
    __shared__ extern uint8_t temp[];
    const int tileW = BLOCK + 2 * offset;
    const int tileH = BLOCK + 2 * offset;
    const int stride = tileW;

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t bx = blockIdx.x * BLOCK;
    const uint32_t by = blockIdx.y * BLOCK;

    const uint32_t srcX = bx + tx;
    const uint32_t srcY = by + ty;

    for (int y = ty; y < tileH; y += BLOCK) {
        for (int x = tx; x < tileW; x += BLOCK) {
            int gx = (int)bx + x - offset;
            int gy = (int)by + y - offset;

            gx = min(max(gx, 0), (int)width - 1);
            gy = min(max(gy, 0), (int)height - 1);

            int idx = y * stride + x;
            temp[idx] = buf[(size_t)gx + pitch * (size_t)gy];
        }
    }

    __syncthreads();

    if (srcX >= width || srcY >= height) return;

    const int pox = tx + offset; // plus offset
    const int poy = ty + offset;
    const int mox = tx - offset; // minus offset
    const int moy = ty - offset;

    uint16_t sum = 0;
    sum += temp[pox + stride * poy];      // left
    sum += temp[pox + stride * moy];      // right
    sum += temp[mox + stride * poy];      // top
    sum += temp[moy + stride * moy];      // bottom

    uint8_t avg = sum >> 2;

    buf[srcX + pitch * srcY] = avg;
}

__global__ void unsharp(int strength, uint8_t* buf, size_t pitch, uint8_t* origin, uint32_t width, uint32_t height)
{
    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t bx = blockIdx.x * blockDim.x;
    const uint32_t by = blockIdx.y * blockDim.y;

    uint32_t srcX = bx + tx;
    uint32_t srcY = by + ty;
    if (srcX >= width || srcY >= height) return;
    int oColor = (int)origin[srcX + pitch * srcY];
    int bColor = (int)buf[srcX + pitch * srcY];

    int diff = oColor - bColor;
    int sharped = oColor + (strength * diff) / 100;

    buf[srcX + pitch * srcY] = (uint8_t)min(max(sharped, 0), 255);
}