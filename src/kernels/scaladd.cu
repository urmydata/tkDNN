#include "kernels.h"
#include <math.h>

__global__ void scal_add_kernel(dnnType* dstData, int size, float alpha, float beta, int inc)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < size) dstData[i*inc] = dstData[i*inc] * alpha + beta;
}

__global__ void scal_add_masks_kernel(dnnType* dstData, int size, float alpha, float beta, int inc, int mask_offset)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	int array_index = blockIdx.z * (mask_offset * gridDim.y) +  blockIdx.y * mask_offset;
	//int array_index = blockIdx.y * mask_index;

    if (i < size)  dstData[array_index + i*inc] = dstData[array_index + i*inc] * alpha + beta;
}

void scalAdd(dnnType* dstData, int size, float alpha, float beta, int inc, cudaStream_t stream)
{
    int blocks = (size+255)/256;
    int threads = 256;
    
    scal_add_kernel<<<blocks, threads, 0, stream>>>(dstData, size, alpha, beta, inc);
}

void scalAddMasks(dnnType* dstData, int batch, int size, float alpha, float beta, int inc, int n_masks, int mask_offset, cudaStream_t stream)
{
 	dim3 dimBlock((size+255)/256, n_masks, batch);
    //int blocks = (size+255)/256;
    int threads = 256;
   
	//printf("merong: %d, %d\n", size, (size+255)/256);

    scal_add_masks_kernel<<<dimBlock, threads, 0, stream>>>(dstData, size, alpha, beta, inc, mask_offset);
}

