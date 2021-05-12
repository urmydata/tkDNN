#include "kernels.h"

__global__
void activation_logistic(dnnType *input, dnnType *output, int size) {

    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<size) {    
        output[i] =  1.0f/(1.0f + exp(-input[i]));;
    }
 }


__global__
void activation_logistic_masks(dnnType *input, dnnType *output, int size, int offset, int mask_offset) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
	//int array_index = blockIdx.y * mask_index  + offset;
	int array_index = blockIdx.z * (mask_offset * gridDim.y) +  blockIdx.y * mask_offset + offset;

    if(i<size) {    
        output[array_index+i] =  1.0f/(1.0f + exp(-input[array_index + i]));;
    }
 }

/**
    LOGISTIC activation function
*/
void activationLOGISTICForward(dnnType* srcData, dnnType* dstData, int size, cudaStream_t stream)
{
    int blocks = (size+255)/256;
    int threads = 256;
    
    activation_logistic<<<blocks, threads, 0, stream>>>(srcData, dstData, size);
}

void activationLOGISTICForwardMasks(dnnType* srcData, dnnType* dstData, int batch, int size, int offset, int n_masks, int mask_offset, cudaStream_t stream)
{
 	dim3 dimBlock((size+255)/256, n_masks, batch);
    //int blocks = (size+255)/256;
    int threads = 256;
    
    activation_logistic_masks<<<dimBlock, threads, 0, stream>>>(srcData, dstData, size, offset, mask_offset);
}




