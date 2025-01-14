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
extern __constant__ HashParams c_hashParams;

struct PointXYZINormal {
    float x, y, z;
    float intensity;
    float normal_x, normal_y, normal_z;
};

bool loadPCDFile(const std::string& filename, std::vector<PointXYZINormal>& points) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    std::string line;
    bool header = true;
    while (std::getline(file, line)) {
        if (header) {
            if (line == "DATA ascii") {
                header = false;
            }
            continue;
        }

        std::istringstream iss(line);
        PointXYZINormal point;
        if (!(iss >> point.x >> point.y >> point.z >> point.intensity >> point.normal_x >> point.normal_y >> point.normal_z)) {
            break;
        }
        points.push_back(point);
    }

    file.close();
    return true;
}

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

__global__ void extract_ply(HashData* hash,float3* d_voxels,float* count)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=2000000*10) return;
    if(hash->d_hash[idx].ptr!=FREE_ENTRY)
    {
        //printf("1111\n");
        for(int i=0;i<512;i++)
        {
            //printf("2222\n");
            if(hash->d_SDFBlocks[hash->d_hash[idx].ptr+i].weight_sum!=0)
            {
                uint3 uvoxellocalpos = hash->delinearizeVoxelIndex(i);
                int3 voxellocalpos = make_int3(uvoxellocalpos.x,uvoxellocalpos.y,uvoxellocalpos.z);
                int3 voxelpos = voxellocalpos+hash->d_hash[idx].pos;
                float3 worldpos = hash->virtualVoxelPosToWorld(voxelpos);
                float a = atomicAdd(count,1.0f) + 0.001;
                //printf("%f\n",*count);
                d_voxels[(int)a]=worldpos;
            }
        }
    }
}

void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

int main() {
    std::vector<PointXYZINormal> cloud;
    if (!loadPCDFile("/home/zzz/code/hash/point_cloud_7377_points.pcd", cloud)) {
        std::cerr << "Failed to read PCD file" << std::endl;
        return -1;
    }

    std::cout << "Point cloud size: " << cloud.size() << std::endl;
    size_t num_points = cloud.size();
    float3* host_points = new float3[num_points];
    float3* host_normals = new float3[num_points];

    // 使用更快的方法计算法线
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    for (size_t j = 0; j < num_points; ++j) {
        pcl::PointXYZ point;
        point.x = cloud[j].x;
        point.y = cloud[j].y;
        point.z = cloud[j].z;
        cloud_ptr->points.push_back(point);
    }

    tree->setInputCloud(cloud_ptr);
    ne.setInputCloud(cloud_ptr);
    ne.setSearchMethod(tree);
    ne.setKSearch(5);
    ne.compute(*cloud_normals);

    for (size_t i = 0; i < num_points; ++i) {
        host_points[i] = make_float3(cloud[i].x, cloud[i].y, cloud[i].z);
        host_normals[i] = make_float3(cloud_normals->points[i].normal_x, cloud_normals->points[i].normal_y, cloud_normals->points[i].normal_z);
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
    // Marching Cubes to extract mesh
    MarchingCubesParams mcParams = CUDAMarchingCubesHashSDF::parametersFromGlobalAppState(10000000, 0, 0.05, 2000000);
    CUDAMarchingCubesHashSDF marchingCubes(mcParams);
    
    marchingCubes.extractIsoSurface(hash, vec3f(0.0f, 0.0f, 0.0f), vec3f(1.0f, 1.0f, 1.0f), false);
    cudaDeviceSynchronize();
    marchingCubes.export_ply("output_mesh.ply");

    hash.free();
    cudaMemcpy(&count,d_count,sizeof(int),cudaMemcpyDeviceToHost);    
    std::cout<<"count: "<<count<<std::endl;
    
    // cudaFree(d_hashdata);
    // cudaFree(device_points);
    // cudaFree(device_normals);
    // cudaFree(d_count);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    // }
    return 0;
}