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

	cudaEventRecord(start, 0);
	saxpy<<<blockNum, blockSize>>>(arrSize, a, dX, dY);
	cudaEventRecord(stop, 0);
	cudaMemcpy(Y, dY, byteSize, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milSecs = 0.0;
	cudaEventElapsedTime(&milSecs, start, stop);

	float maxError = 0.0;
	for(i = 0; i < arrSize; i++)
		maxError = max(maxError, abs(Y[i]-30.0));
	printf("Max Error: %f\n", maxError);
	//3: reading of X and reading and writing of Y
	printf("Effective Bandwidth[GB/s]: %f\n", 1e-6*sizeof(float)*arrSize*3 / milSecs);

	// compute theoretical peak bandwidth
	int nDevs;
	cudaGetDeviceCount(&nDevs);
	for(i = 0; i < nDevs; ++i)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d", i);
		printf("  Device Name: %s", prop.name);
		printf("  Memory Clock Rate[Mhz]: %d\n", prop.memoryClockRate);
		printf("  Memory Bus Width[bits]: %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth[GB/s]: %f\n",
		 1.0e-3*2*prop.memoryClockRate*prop.memoryBusWidth/8);
	}


	cudaFree(dX);
	cudaFree(dY);
	free(X);
	free(Y);

	return 0;

}