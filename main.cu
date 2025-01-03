#include<iostream>
#include"Voxelhash.h"
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<CUDAHashParams.h>
__global__ void test(HashData * hash,int * d_count)
{
    int a=56*blockIdx.x;
    int b=56*blockIdx.x;
    int c=56*blockIdx.x;
    // for(int i=1;i<10;i+=8)
    // {
    //     int a = 56*i;
    //     int b= 56*i;
    //     int c=56*i;
    //     printf("%d %d %d\n",a,b,c);
    //     const int p0 = 73856093;
    //     const int p1 = 19349669;
    //     const int p2 = 83492791;
    //     int3 voxelpos=make_int3(a,b,c);
    //     int3 sdfpos=hash->virtualVoxelPosToSDFBlock(voxelpos);
    //     int pos=hash->computeHashPos(sdfpos);
    //     printf("%d\n",pos);
    //     hash->allocBlock(sdfpos);
    //     HashEntry entry=hash->getHashEntryForSDFBlockPos(sdfpos);
    //     printf("%d\n\n",entry.ptr);   
    // }
    //printf("%d %d %d\n",a,b,c);
    const int p0 = 73856093;
    const int p1 = 19349669;
	const int p2 = 83492791;
    int3 voxelpos=make_int3(a,b,c);
    int3 sdfpos=hash->virtualVoxelPosToSDFBlock(voxelpos);
    int pos=hash->computeHashPos(sdfpos);
    //printf("%d\n",pos);
    //printf("%d\n",hash->d_hash[pos].ptr);
    hash->allocBlock(sdfpos);
    HashEntry entry=hash->getHashEntryForSDFBlockPos(sdfpos);
    if(entry.ptr==FREE_ENTRY)
        atomicAdd(&d_count[0], 1);
    //printf("%d\n",entry.ptr);
    return;
}
__global__ void test1(HashData *hash, int hashNumBuckets, int hashBucketSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= hashNumBuckets * hashBucketSize)
    {
        return;
    }
    if(hash->d_hash[idx].ptr!=FREE_ENTRY)
        printf("%d\n",idx);
}
    // printf("%d %d %d\n",a,b,c);
    // printf("%d %d %d\n",sdfpos.x,sdfpos.y,sdfpos.z);
    // unsigned int pos=hash.computeHashPos(sdfpos);
    // printf("%d\n",((100*p0)^(100*p1)^(100*p2))%c_hashParams.m_hashNumBuckets);
    // printf("%d\n",pos);
int main()
{
    HashData hash;
    HashParams params;
    params.m_hashBucketSize=100;
    params.m_hashMaxCollisionLinkedListSize=10;
    params.m_hashNumBuckets=1000;
    params.m_integrationWeightMax=0.5;
    params.m_integrationWeightSample=0.5;
    params.m_maxIntegrationDistance=10;
    params.m_numOccupiedBlocks=0;
    params.m_numSDFBlocks=100000;
    params.m_SDFBlockSize=8;
    params.m_truncation=10;
    params.m_truncScale=10;
    params.m_virtualVoxelSize=0.1;
    hash.allocate(params,true);
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
    // int numThreads = 256;  // 线程数目
    // int numBlocks = (params.m_hashNumBuckets * params.m_hashBucketSize + numThreads - 1) / numThreads;
	// test1<<<numBlocks, numThreads>>>(d_hashdata, params.m_hashNumBuckets, params.m_hashBucketSize);
    // cudaDeviceSynchronize();  // 确保内核执行完成
    hash.free();
    cudaMemcpy(&count,d_count,sizeof(int),cudaMemcpyDeviceToHost);    
    std::cout<<"最终count为:"<<count<<std::endl;
    cudaFree(d_hashdata);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }
    return 0;
    //printf("%d\n",sizeof(HashEntry));
    pointcloudtohashvoxel();
}