#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

inline
cudaError_t checkCuda(cudaError_t result)
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

template <typename T>
void runTest(int nMB)
{
	const int blockSize = 256;
	const unsigned int arrSize = nMB*1024*1024 / sizeof(T);

	int i;
	T *da;
	float milSecs;
	cudaEvent_t start, stop;
	//33: for stride access case
	checkCuda(cudaMalloc((void**)&da, 33*arrSize*sizeof(T)));
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	offset<<<arrSize/blockSize, blockSize>>>(da, 0);//warm up
	printf("Offset: Bandwidth[GB/s]\n");
	for(i = 0; i <=32; ++i)
	{
		cudaEventRecord(start, 0);
		cudaMemset(da, 0, arrSize*sizeof(T));
		offset<<<arrSize/blockSize, blockSize>>>(da, i);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milSecs, start, stop);
		printf(" %d, %f\n", i, 2*nMB/milSecs);
	}

	stride<<<arrSize/blockSize, blockSize>>>(da, 1);//warm up
	printf("Stride: Bandwidth[GB/s]\n");
	for(i = 1; i <= 32; ++i)
	{
		cudaEventRecord(start, 0);
		cudaMemset(da, 0, arrSize*sizeof(T));
		stride<<<arrSize/blockSize, blockSize>>>(da, i);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milSecs, start, stop);
		printf(" %d, %f\n", i, 2*nMB/milSecs);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(da);
}

int main(int argc, char** argv)
{
	if(argc != 2)
	{
		fprintf(stderr, "Provide size of transfer in MB!\n");
		exit(EXIT_FAILURE);
	}
	int nMB = atoi(argv[1]);

	printf("Transfer size of %d.\n", nMB);
	printf("Results for double precision:\n");
	runTest<double>(nMB);
	printf("Results for single precision:\n");
	runTest<float>(nMB);

	return 0;
}