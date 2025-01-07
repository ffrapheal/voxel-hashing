#include <iostream>
#include "VoxelHash.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
__global__ void test(HashData * hash,int * count,float * pos,int * num_points)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=*num_points) return;
    float3 worldpos=make_float3(pos[idx*3],pos[idx*3+1],pos[idx*3+2]);
    //assume (a b c) is the coordinate of a point.
    bool insert;
    int3 voxelpos = hash->worldToVirtualVoxelPos(worldpos);
    uint hashpos=hash->computeHashPos(worldpos);
    insert=hash->insertHashEntryElement(worldpos);
    __threadfence();
    HashEntry curr = hash->getHashEntry(worldpos);
    uint h = hash->computeHashPos(worldpos);
    if(curr.ptr!=FREE_ENTRY)
    {
        atomicAdd(&count[0], 1);
        Voxel v = hash->getVoxel(worldpos);
    }
    //printf("a:%d b:%d c:%d pos:%d ptr:%d insert:%s\n",voxelpos.x,voxelpos.y,voxelpos.z,pos,curr.ptr,insert ? "true" : "false");
    return;
}

void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

int main() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 读取PCD文件
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/zzz/code/hash/point_cloud_7377_points.pcd", *cloud) == -1) // 这里替换为你的PCD文件路径
    {
        PCL_ERROR("无法读取文件 \n");
        return -1;
    }
    // 输出点云信息
    std::cout << "点云宽度: " << cloud->width << std::endl;
    std::cout << "点云高度: " << cloud->height << std::endl;
    std::cout << "点云大小: " << cloud->points.size() << std::endl;
    // 遍历点云中的每个点
    size_t num_points = cloud->points.size();
    float* host_points = new float[num_points * 3]; // 每个点有三个坐标
    for (size_t i = 0; i < num_points; ++i) {
        host_points[i * 3] = cloud->points[i].x;
        host_points[i * 3 + 1] = cloud->points[i].y;
        host_points[i * 3 + 2] = cloud->points[i].z;
    }

    // 分配GPU内存
    float* device_points;
    cudaMalloc(&device_points, num_points * 3 * sizeof(float));
    int* device_num_points;
    cudaMalloc(&device_num_points,sizeof(int));
    // 复制数据到GPU
    cudaMemcpy(device_points, host_points, num_points * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_num_points, &num_points, sizeof(int), cudaMemcpyHostToDevice);
    delete[] host_points;
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
    dim3 blockSize(1024);
    dim3 gridSize((num_points + blockSize.x - 1) / blockSize.x);
    test<<<gridSize,blockSize>>>(d_hashdata,d_count,device_points,device_num_points);
    cudaDeviceSynchronize();
    hash.free();
    cudaMemcpy(&count,d_count,sizeof(int),cudaMemcpyDeviceToHost);    
    std::cout<<"count: "<<count<<std::endl;
    cudaFree(d_hashdata);
    cudaFree(device_points);
    cudaFree(d_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }
    return 0;
}