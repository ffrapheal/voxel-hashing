#include<iostream>
#include"Voxelhash.h"
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
__global__ void test(struct HashData hash)
{
    int a=threadIdx.x;
    int b=threadIdx.y;
    int c=blockIdx.x;
    int3 t=make_int3(a,b,c);
    struct Voxel v1{0,0,0.5};
    hash.setVoxel(t,v1);
    return;
}
int main()
{
    struct HashData hash;
    struct HashParams params;
    params.m_hashBucketSize=4;
    params.m_hashMaxCollisionLinkedListSize=10;
    params.m_hashNumBuckets=10;
    params.m_integrationWeightMax=0.5;
    params.m_integrationWeightSample=0.5;
    params.m_maxIntegrationDistance=10;
    params.m_numOccupiedBlocks=0;
    params.m_numSDFBlocks=100;
    params.m_SDFBlockSize=4*4*4;
    params.m_truncation=10;
    params.m_truncScale=10;
    params.m_virtualVoxelSize=0.1;
    hash.allocate(params,true);
    dim3 blockSize(4,4);
    dim3 gridSize(8);
    hash.free();
    return 0;
}