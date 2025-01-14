#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include "VoxelHash.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <eigen3/Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>
#include "CUDAMarchingCubesHashSDF.h"
#include <yaml-cpp/yaml.h>
#include <pcl/io/pcd_io.h>
#include <unordered_map>
#include "pointpreprocessing.h"
std::unordered_map<int, int3> map;
#define hash_max 100000000


// int count =0;

// void pointpreprocessing(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float voxel_size)
// {
//     // int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     // if (idx >= cloud.size()) return;
//     // float3 world_point = make_float3(cloud[idx].x, cloud[idx].y, cloud[idx].z);
//     // HashData hash_data;


//     // int3 index = worldtovirualvoxelindex(world_point, voxel_size);
//     // point_blocks[idx].block_index = index.x;
//     // point_blocks[idx].point_index = idx;
//     // test hash colision
//     for(int i = 0; i < cloud->size(); i++){
//         int3 virtualVoxelPos = worldToVirtualVoxelPos(make_float3(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z));
//         int a = computeHashPos(virtualVoxelPos);
//         printf("a: %d\n",a);
//         printf("virtualVoxelPos: %d %d %d\n",virtualVoxelPos.x,virtualVoxelPos.y,virtualVoxelPos.z);
//         if(map.find(a) != map.end()){
//             if(map[a].x!=virtualVoxelPos.x || map[a].y!=virtualVoxelPos.y || map[a].z!=virtualVoxelPos.z){
//                 count++;
//             }
//         }
//         else
//         {
//             map[a] = virtualVoxelPos;
//         }
//     }

//     std::cout << "hash colision count: " << count << std::endl;
// }

int main() {

    // 生成10000个随机点
    const int num_points = 10000;
    float3* host_points = new float3[num_points];
    float3* host_normals = new float3[num_points];

    // 设置平面参数 (假设是xz平面,y=1)
    float3 plane_normal = make_float3(0.0f, 1.0f, 0.0f);
    float plane_d = 1.0f;

    // 随机数生成器
    srand(time(NULL));

    // 生成随机点和法向量
    for(int i = 0; i < num_points; i++) {
        // 在xz平面上随机生成点
        float x = (float)rand() / RAND_MAX * 2.0f - 1.0f; // [-1,1]范围
        float z = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        
        // 添加高斯噪声到y坐标
        float gaussian_noise = 0;
        for(int j = 0; j < 12; j++) {
            gaussian_noise += ((float)rand() / RAND_MAX - 0.5f);
        }
        gaussian_noise *= 0.02f; // 缩放噪声
        
        float y = plane_d + gaussian_noise;
        
        host_points[i] = make_float3(x, y, z);
        host_normals[i] = plane_normal;
    }

    // 分配GPU内存
    float3* device_points;
    float3* device_normals;
    cudaMalloc(&device_points, num_points * sizeof(float3));
    cudaMalloc(&device_normals, num_points * sizeof(float3));

    // 将数据复制到GPU
    cudaMemcpy(device_points, host_points, num_points * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(device_normals, host_normals, num_points * sizeof(float3), cudaMemcpyHostToDevice);

    // 释放主机内存
    delete[] host_points;
    delete[] host_normals;
    
    HashData hash;
    hash.allocate(true);

    HashData *d_hashdata;

    float count=0;
    float *d_count;

    cudaMalloc(&d_hashdata,sizeof(HashData));
    cudaMalloc(&d_count,sizeof(float));

    cudaMemcpy(d_hashdata,&hash,sizeof(HashData),cudaMemcpyHostToDevice);
    cudaMemcpy(d_count,&count,sizeof(float),cudaMemcpyHostToDevice);

    dim3 blockSize(1024);
    dim3 gridSize((num_points + blockSize.x - 1) / blockSize.x);

    // test<<<gridSize,blockSize>>>(d_hashdata,d_count,device_points,num_points);
    updatesdfframe<<<gridSize,blockSize>>>(d_hashdata,device_points,device_normals,num_points);
    cudaDeviceSynchronize();
    updatesdfframe<<<gridSize,blockSize>>>(d_hashdata,device_points,device_normals,num_points);
    cudaDeviceSynchronize();
    updatesdfframe<<<gridSize,blockSize>>>(d_hashdata,device_points,device_normals,num_points);
    cudaDeviceSynchronize();
    // Marching Cubes to extract mesh
    MarchingCubesParams mcParams = CUDAMarchingCubesHashSDF::parametersFromGlobalAppState(10000000, 0, 0.05, 2000000);
    CUDAMarchingCubesHashSDF marchingCubes(mcParams);
    
    marchingCubes.extractIsoSurface(hash, vec3f(0.0f, 0.0f, 0.0f), vec3f(1.0f, 1.0f, 1.0f), false);
    cudaDeviceSynchronize();
    marchingCubes.export_ply("output_mesh.ply");

    hash.free();
    cudaMemcpy(&count,d_count,sizeof(int),cudaMemcpyDeviceToHost);    
    // std::cout<<"count: "<<count<<std::endl;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }

    return 0;
}