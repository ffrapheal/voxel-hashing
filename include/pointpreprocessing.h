#ifndef POINTPREPROCESSING
#define POINTPREPROCESSING
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <VoxelHash.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#define point_block_size 1000
#define hash_max 100000000

struct point_block
{
    int block_index;
    int point_index;
    bool operator<(const point_block& other) const {
        return block_index < other.block_index;
    }
};

struct point_count
{
    int count;
    float3 worldpos;
    point_count(int count, float3 worldpos) : count(count), worldpos(worldpos) {}
};

// because in the test the voxel size has not determined, so we use 0.05 as the voxel size.
__device__
int3 worldToVirtualVoxelPos(const float3& pos) {
	const float3 p = pos / 0.5;
	return make_int3(p + make_float3(sign(p)) * 0.5f);
}

__device__
uint computeHashPos(const int3& virtualVoxelPos) {    
	const int p0 = 73856093;
	const int p1 = 19349669;
	const int p2 = 83492791;
	int res = ((virtualVoxelPos.x * p0) ^ (virtualVoxelPos.y * p1) ^ (virtualVoxelPos.z * p2)) % hash_max;
	if (res < 0) res += hash_max;
	return (uint)res;
}

__global__
void pointpreprocessing(point_count *device_points, point_block *point_blocks, int numPoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    float3 world_point = device_points[idx].worldpos;
    int3 virtualVoxelPos = worldToVirtualVoxelPos(world_point);
    uint hash_pos = computeHashPos(virtualVoxelPos);
    point_blocks[idx].block_index = hash_pos;
    point_blocks[idx].point_index = idx;
}

__host__
float3 computeNormal(const std::vector<point_count> points,int start_index,int end_index,point_block* host_points) {
    // 至少需要3个点才能计算法线
    if (end_index - start_index < 3) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    // 将点转换为Eigen矩阵
    Eigen::MatrixXf points_matrix(end_index - start_index, 3);
    for (size_t i = start_index; i < end_index; i++) {
        points_matrix(i - start_index, 0) = points[host_points[i].point_index].worldpos.x;
        points_matrix(i - start_index, 1) = points[host_points[i].point_index].worldpos.y;
        points_matrix(i - start_index, 2) = points[host_points[i].point_index].worldpos.z;
    }
    // 计算质心
    Eigen::Vector3f centroid = points_matrix.colwise().mean();
    
    // 构建协方差矩阵
    Eigen::MatrixXf centered = points_matrix.rowwise() - centroid.transpose();
    Eigen::Matrix3f covariance = (centered.transpose() * centered) / float(points_matrix.rows() - 1);
    
    // 计算特征值和特征向量
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
    Eigen::Vector3f normal = eigen_solver.eigenvectors().col(0) ;


    // 转换回float3
    return make_float3(normal.x(), normal.y(), normal.z());
}

__host__
std::vector<float3> host_pointpreprocessing(std::vector<point_count>& point_counts,std::vector<float3>& normals)
{
    // read pcd file from new frame.
    std::vector<float3> valid_points;
    std::vector<point_count> next_points;
    int thereshold = 5;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/zzz/code/hash/point_cloud_7377_points.pcd", *cloud) == -1)
    {
        std::cout << "无法读取点云文件!" << std::endl;
        return valid_points;
    }
    std::cout << "成功读取点云，点数: " << cloud->size() << std::endl;
    
    // merge new frame data and old data.
    std::vector<point_count> points;
    points.reserve(cloud->size());
    for(int i = 0; i < cloud->size(); i++) {
        points.push_back(point_count(1,make_float3(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z)));
    }
    for(int i = 0; i < point_counts.size(); i++) {
        if(point_counts[i].count + 1 > 5) {
            continue;
        }
        points.push_back(point_count(point_counts[i].count+1, point_counts[i].worldpos));
    }

    // copy points to device.
    point_count* device_points;
    point_block* device_point_blocks;
    cudaMalloc(&device_points, points.size() * sizeof(point_count));
    cudaMalloc(&device_point_blocks, points.size() * sizeof(point_block));
    cudaMemcpy(device_points, points.data(), points.size() * sizeof(point_count), cudaMemcpyHostToDevice);
    
    dim3 block_size(1024, 1, 1);
    dim3 grid_size((points.size() + block_size.x - 1) / block_size.x, 1, 1);
    pointpreprocessing<<<grid_size, block_size>>>(device_points, device_point_blocks, points.size());
    cudaDeviceSynchronize();
    // sort the point_blocks using thrust
    thrust::device_ptr<point_block> thrust_point_blocks = thrust::device_pointer_cast(device_point_blocks);
    thrust::sort(thrust_point_blocks, thrust_point_blocks + points.size());
    
    // copy back to host
    point_block* host_point_blocks = new point_block[points.size()];
    cudaMemcpy(host_point_blocks, device_point_blocks, points.size() * sizeof(point_block), cudaMemcpyDeviceToHost);
    std::vector<int> block_change_indices;
    block_change_indices.push_back(0);
    
    for(int i = 1; i < points.size(); i++) {
        if(host_point_blocks[i].block_index != host_point_blocks[i-1].block_index) {
            block_change_indices.push_back(i);
        }
    }
    int first_block_size = block_change_indices[0];
    if(first_block_size >= thereshold) {
        float3 normal = computeNormal(points,0,block_change_indices[0],host_point_blocks);
        for(int j = 0; j < block_change_indices[0]; j++) {
            valid_points.push_back(points[host_point_blocks[j].point_index].worldpos);
            normals.push_back(normal);
        }
    } else {
        for(int j = 0; j < block_change_indices[0]; j++) {
            next_points.push_back(points[host_point_blocks[j].point_index]);
        }
    }
    
    // 处理其余区块
    for(int i = 0; i < block_change_indices.size()-1; i++) {
        int block_size = block_change_indices[i+1] - block_change_indices[i];
        if(block_size >= thereshold) {
            float3 normal = computeNormal(points,block_change_indices[i],block_change_indices[i+1],host_point_blocks);
            // 将连续5个以上的点加入filtered_points
            for(int j = block_change_indices[i]; j < block_change_indices[i+1]; j++) {
                valid_points.push_back(points[host_point_blocks[j].point_index].worldpos);
                normals.push_back(normal);
            }
        } else {
            // 将少于5个连续点的加入remaining_points
            for(int j = block_change_indices[i]; j < block_change_indices[i+1]; j++) {
                next_points.push_back(points[host_point_blocks[j].point_index]); 
            }
        }
    }
    
    // 处理最后一个区块
    int last_block_start = block_change_indices[block_change_indices.size()-1];
    int last_block_size = points.size() - last_block_start;
    if(last_block_size >= thereshold) {
        float3 normal = computeNormal(points,last_block_start,points.size(),host_point_blocks);
        for(int j = last_block_start; j < points.size(); j++) {
            valid_points.push_back(points[host_point_blocks[j].point_index].worldpos);
            normals.push_back(normal);
        }
    } else {
        for(int j = last_block_start; j < points.size(); j++) {
            next_points.push_back(points[host_point_blocks[j].point_index]);
        }
    }
    
    // 更新输出的point_counts
    point_counts = next_points;
    // 清理内存
    delete[] host_point_blocks;
    cudaFree(device_points);
    cudaFree(device_point_blocks);
    
    return valid_points;
}

#endif  