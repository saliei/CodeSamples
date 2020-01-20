#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if(result != cudaSuccess)
		fprintf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
	assert(result == cudaSuccess);
#endif
	return result;
}

__global__
void arrAdd(float* arr, int offset)
{
	int idx = offset + threadIdx.x + blockIdx.x*blockDim.x;
	arr[idx] += 1.0;
}

float maxError(float *A, unsigned int n)
{
	float maxErr = 0.0;
	unsigned int i;
	for(i = 0; i < n; ++i)
		maxErr = max(maxErr, abs(A[i]-1.0));
	return maxErr;
}

int main(int argv, char** argv)
{
	const int numStrms = 4, blockSize = 256;
	const int n = 4 * 1024 * numStrms * blockSize;
	const int strmSize = n / numStrms;
	const int strmBytes = strmSize * sizeof(float);
	const int bytes = n * sizeof(float);

	//allocate pinned host memory and device memory
	float *A, *dA;
	checkCuda(cudaMallocHost((void**)&A, bytes));
	checkCuda(cudaMalloc((void**)&dA, bytes));

	cudaEvent_t stop, start;
	cudaStream_t stream[numStrms];
	float milSecs;
	int i;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	for(i = 0; i < numStrms; i++)
		cudaStreamCreate(&stream[i]);

	//base case for sequential transfer and execution
	memset(A, 0, bytes);
	cudaEventRecord(start, 0);
	cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
	arrAdd<<<n/blockSize, blockSize>>>(dA, 0);
	cudaMemcpy(A, dA, bytes, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milSecs, start, stop);
	printf("Time for sequential: %f", milSecs);
	printf("  max error: %f\n", maxError(A, n));

	//async v1: big loop
	memset(A, 0, bytes);
	cudaEventRecord(start, 0);
	for(i = 0; i < numStrms; ++i)
	{
		offset = i * strmSize;
		cudaMemcpyAsync(&dA[offset], &A[offset], strmBytes, cudaMemcpyHostToDevice, &stream[i]);
		arrAdd<<<strmSize/blockSize, blockSize>>>(dA, offset);
		cudaMemcpyAsync(&A[offset], &dA[offset], strmBytes, cudaMemcpyDeviceToHost, &stream[i]);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milSecs, start, stop);
	printf("Time for async v1: %f", milSecs);
	printf("  max error: %f\n", maxError(A, n));

	//async v2: unrolled loops
	memset(A, 0, bytes);
	cudaEventRecord(start, 0);
	for(i = 0; i < numStrms; ++i)
	{
		offset = i * strmSize;
		cudaMemcpyAsync(&dA[offset], &A[offset], strmBytes, cudaMemcpyHostToDevice, &stream[i]);
	}
	for(i = 0; i < numStrms; ++i)
	{
		offset = i * strmSize;
		arrAdd<<<strmSize/blockSize, blockSize>>>(dA, offset);
	}
	for(i = 0; i < numStrms; ++i)
	{
		offset = i * strmSize;
		cudaMemcpyAsync(&A[offset], &dA[offset], strmSize, cudaMemcpyDeviceToHost, &stream[i]);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milSecs, start, stop);
	printf("Time for async v2: %f\n", milSecs);
	printf("  max error: %f\n", maxError(A, n));

	//cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	for(i = 0; i < numStrms; ++i)
		cudaStreamDestroy(stream[i]);
	cudaFreeHost(A);
	cudaFree(dA);

	return 0;
}