#include <iostream>
#include <math.h>

const int N = 1 << 20;
const float PI = 3.14159;

__global__
void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = tid; i < n; i += blockDim.x * gridDim.x)
        x[i] = sqrt(pow(PI, i));
}

int main(int argc, char **argv)
{
    const int num_streams = 8;

    cudaStream_t streams[num_streams];
    float *data[num_streams];

    for(int i = 0; i < num_streams; i++)
    {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&data[i], N*sizeof(float));
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        //launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceReset();

    return 0;
}
