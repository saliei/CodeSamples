#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

inline
cudaError_t checkCud(cudaError_t result)
{
#if defined (DEBUG) || defined (_DEBUG)
	if(result != cudaSuccess)
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
	assert(result == cudaSuccess);
#endif
	return result;
}

template <typename T>
__global__ void offset(T *arr, int s)
{
	int idx = s + threadIdx.x + blockDim.x*blockIdx.x;
	arr[idx] += 1.0;
}

template <typename T>
__global__ void stride(T *arr, int s)
{
	int idx = s * (threadIdx.x + blockDim.x*blockIdx.x);
	arr[idx] += 1.0;
}

