#include<iostream>
#include"VoxelHash.h"
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void test(HashData * hash,int * d_count)
{
    float a=56*blockIdx.x;
    float b=56*blockIdx.x;
    float c=56*blockIdx.x;
    float3 worldpos=make_float3(a,b,c);
    uint pos=hash->computeHashPos(worldpos);
    return;
}
int main()
{
    HashData hash;
    HashParams params;
    hash.allocate(true);
    HashData *d_hashdata;
    int count=0;
    int *d_count;
    cudaMalloc(&d_hashdata,sizeof(HashData));
    cudaMalloc(&d_count,sizeof(int));
    cudaMemcpy(d_hashdata,&hash,sizeof(HashData),cudaMemcpyHostToDevice);
    cudaMemcpy(d_count,&count,sizeof(int),cudaMemcpyHostToDevice);
    dim3 blockSize(1,1);
    dim3 gridSize(1000);
    test<<<gridSize,blockSize>>>(d_hashdata,d_count);
    cudaDeviceSynchronize();
    hash.free();
    cudaMemcpy(&count,d_count,sizeof(int),cudaMemcpyDeviceToHost);    
    std::cout<<"count: "<<count<<std::endl;
    cudaFree(d_hashdata);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }
    return 0;
}