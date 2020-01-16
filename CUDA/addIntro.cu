#include <stdio.h>
#include <stdlib.h>

__global__
void add(double* A, double* B, double* C, unsigned int arrSize)
{
	unsigned int index = blockDim.x*blockId.x + threadId.x;
	unsigned int stride = gridDim.x * blockDim.x;
	unsigned int i;
	for(i = index; i < arrSize; i += stride)
		C[index] = A[index] + B[index];
}

int main(int argc, char** argv)
{
	if(argc != 2)
	{
		fprintf(stderr, "Array size must be provided.\n");
		exit(EXIT_FAILURE);
	}
	unsigned int arrSize = atoi(argv[1]);
	byteSize = sizeof(double) * arrSize;

	double* A = (double*)malloc(byteSize);
	double* B = (double*)malloc(byteSize);
	double* C = (double*)malloc(byteSize);

	srand(time(NULL));
	unsigned int i;
	for(i = 0; i < arrSize; ++i)
	{
		A[i] = 1.0;
		B[i] = 2.0;
		C[i] = 0;
	}

	double* d_A, d_B, d_C;
	cudaMalloc((double**)&d_A, byteSize);
	cudaMalloc((double**)&d_B, byteSize);
	cudaMalloc((double**)&d_C, byteSize);
	cudaMemcpy(d_A, A, byteSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, byteSize, cudaMemcpyHostToDevice);

	int blockSize = 256;
	int blockNum = (arrSize + blockSize -1) / blockSize;
	
	add<<<blockNum, blockSize>>>(d_A, d_B, d_C, arrSize);
	cudaMemcpy(C, d_C, byteSize, cudaMemcpyDeviceToHost);

	double maxError =0.0;
	for(i = 0; i < arrSize; ++i)
		maxError = max(maxError, abs(C[i]-3.0));
	printf("Max Error: %lf\n", maxError);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(A);
	free(B);
	free(C);

	return 0;
}