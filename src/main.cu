#include <iostream>
#include "VoxelHash.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

__global__ void test(HashData * hash,int * count,float * pos,int num_points)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=num_points) return;
    float3 worldpos=make_float3(pos[idx*3],pos[idx*3+1],pos[idx*3+2]);
    //assume (a b c) is the coordinate of a point.
    int3 voxelpos = hash->worldToVirtualVoxelPos(worldpos);
    uint hashpos=hash->computeHashPos(worldpos);
    hash->insertHashEntryElement(worldpos);
    __threadfence();
    HashEntry curr = hash->getHashEntryForWorldPos(worldpos);
    uint h = hash->computeHashPos(worldpos);
    if(curr.ptr!=FREE_ENTRY)
    {
        atomicAdd(&count[0], 1);
        Voxel * v = hash->getVoxel(worldpos);
        printf("voxelpos: %d %d %d\n",voxelpos.x,voxelpos.y,voxelpos.z);
        printf("voxel: %f %f %f\n",v->sdf_sum,v->weight_sum,v->sdf_sum/v->weight_sum);
    }
    return;
}

void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

int main() {
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
    printf("hello world\n");
    // read pcd file
    if (pcl::io::loadPCDFile<pcl::PointXYZINormal>("/home/zzz/code/hash/point_cloud_7377_points.pcd", *cloud) == -1) // 这里替换为你的PCD文件路径
    {
        PCL_ERROR("failed to read pcd file \n");
        return -1;
    }
    // print point cloud info
    std::cout << "point cloud width: " << cloud->width << std::endl;
    std::cout << "point cloud height: " << cloud->height << std::endl;
    std::cout << "point cloud size: " << cloud->points.size() << std::endl;
    // traverse each point in point cloud
    size_t num_points = cloud->points.size();
    float3* host_points = new float3[num_points]; // 每个点的坐标
    float3* host_normals = new float3[num_points]; // 每个点的法线

    for (size_t i = 0; i < num_points; ++i) {
        host_points[i] = make_float3(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
        host_normals[i] = make_float3(cloud->points[i].normal_x, cloud->points[i].normal_y, cloud->points[i].normal_z);
        printf("point: %f %f %f\n", host_points[i].x, host_points[i].y, host_points[i].z);
        //printf("normal: %f %f %f\n", host_normals[i].x, host_normals[i].y, host_normals[i].z);
    }
    // allocate memory on gpu
    float3* device_points;
    cudaMalloc(&device_points, num_points * sizeof(float3));
    float3* device_normals;  
    cudaMalloc(&device_normals, num_points * sizeof(float3));
    // copy data to gpu
    cudaMemcpy(device_points, host_points, num_points * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(device_normals, host_normals, num_points * sizeof(float3), cudaMemcpyHostToDevice);
    delete[] host_points;
    delete[] host_normals;
    // initialize hash data
    HashData hash;
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

    // test<<<gridSize,blockSize>>>(d_hashdata,d_count,device_points,num_points);
    updatesdfframe<<<gridSize,blockSize>>>(d_hashdata,device_points,device_normals,num_points);
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