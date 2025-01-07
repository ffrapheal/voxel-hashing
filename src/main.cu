#include <iostream>
#include "VoxelHash.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <pcl/point_types.h>
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
    if (pcl::io::loadPCDFile<pcl::PointXYZINormal>("/home/hmy/voxel_hashing_dev/point_cloud_7377_points.pcd", *cloud) == -1) // 这里替换为你的PCD文件路径
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
    float* host_points = new float[num_points * 3]; // each point has three coordinates
    for (size_t i = 0; i < num_points; ++i) {
        host_points[i * 3] = cloud->points[i].x;
        host_points[i * 3 + 1] = cloud->points[i].y;
        host_points[i * 3 + 2] = cloud->points[i].z;
    }

    // allocate memory on gpu
    float* device_points;
    cudaMalloc(&device_points, num_points * 3 * sizeof(float));
    int* device_num_points;
    cudaMalloc(&device_num_points,sizeof(int));
    // copy data to gpu
    cudaMemcpy(device_points, host_points, num_points * 3 * sizeof(float), cudaMemcpyHostToDevice);
    delete[] host_points;


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

    test<<<gridSize,blockSize>>>(d_hashdata,d_count,device_points,num_points);
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