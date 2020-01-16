#include <stdlib.h>
#include <stdio.h>

__global__
void saxpy(int n, float a, float* X, float* Y)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = gridDim.x*blockDim.x;
	int i;
	for(i = index; i < n; i += stride)
		Y[i] = a*X[i] + Y[i];
}

int main(int argc, char** argv)
{
	if(argc != 2)
	{
		fprintf(stderr, "Array size must be provided.\n");
		exit(EXIT_FAILURE);
	}
	const int arrSize = atoi(argv[1]);
	unsigned int byteSize = sizeof(float) * arrSize;

	float a = 10.0;
	float* X = (float*)malloc(byteSize);
	float* Y = (float*)malloc(byteSize);
	
	int i;
	for(i = 0; i < arrSize; i++)
	{
		X[i] = 2.0;
		Y[i] = 10.0;
	}

	float *dX, *dY;
	cudaMalloc((double**)&dX, byteSize);
	cudaMalloc((double**)&dY, byteSize);

	cudaMemcpy(dX, X, byteSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dY, Y, byteSize, cudaMemcpyHostToDevice);

	int blockSize = 256;
	int blockNum = (arrSize+blockSize-1) / blockSize;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	saxpy<<<blockNum, blockSize>>>(arrSize, a, dX, dY);
	cudaEventRecord(stop);
	cudaMemcpy(Y, dY, byteSize, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milSecs = 0.0;
	cudaEventElapsedTime(&milSecs, start, stop);

	float maxError = 0.0;
	for(i = 0; i < arrSize; i++)
		maxError = max(maxError, abs(Y[i]-30.0));
	printf("Max Error: %f\n", maxError);

	cudaFree(dX);
	cudaFree(dY);
	free(X);
	free(Y);

	return 0;

}