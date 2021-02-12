#include <iostream>

#define SIZE 100

__global__ void testAdd(float *a)
{
    for(int i = 0; i < SIZE; i++)
    {
        atomicAdd(&a[i], 1.0f);
    }
}

void cuTestAtomicAdd(float *a)
{
    testAdd<<<1, 10>>>(a);
}

int main(int argc, char **argv)
{
    float *d_data, *h_data;
    h_data = (float*)malloc(SIZE * sizeof(float));
    cudaMalloc((void**)&d_data, SIZE*sizeof(float));
    cudaMemset(d_data, 0, SIZE*sizeof(float));
    cuTestAtomicAdd(d_data);
    cudaMemcpy(h_data, d_data, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < SIZE; i++)
        if(h_data[i] != 10.0f)
            std::cout << "mismatch at " << i << " should be 10.0f, it is: " << h_data[i] << std::endl;
    std::cout << "Success!" << std::endl;

    return 0;

}
