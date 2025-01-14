#include "VoxelHash.h"
#include "CUDAMarchingCubesHashSDF.h"

extern "C" void resetMarchingCubesCUDA(MarchingCubesData& data);
extern "C" void extractIsoSurfaceCUDA(const HashData& hashData, const MarchingCubesParams& params, MarchingCubesData& data);
extern "C" void extractIsoSurfacePass1CUDA(const HashData& hashData, const MarchingCubesParams& params, MarchingCubesData& data);
extern "C" void extractIsoSurfacePass2CUDA(const HashData& hashData, const MarchingCubesParams& params, MarchingCubesData& data, unsigned int numOccupiedBlocks);

#define cudaCheckError(err) if (err != cudaSuccess) { printf("CUDA error: %s, line: %d, file: %s\n", cudaGetErrorString(err), __LINE__, __FILE__); }

void CUDAMarchingCubesHashSDF::create(const MarchingCubesParams& params)
{ 
	m_params = params;
	m_data.allocate(m_params);

	resetMarchingCubesCUDA(m_data);
}

void CUDAMarchingCubesHashSDF::destroy(void)
{
	m_data.free();
}

void CUDAMarchingCubesHashSDF::extractIsoSurface(const HashData& hashData, const vec3f& minCorner, const vec3f& maxCorner, bool boxEnabled)
{
	resetMarchingCubesCUDA(m_data);
	float3 maxc = {maxCorner.x,maxCorner.y,maxCorner.z};
	float3 minc = {minCorner.x,minCorner.y,minCorner.z};
	m_params.m_maxCorner = maxc;
	m_params.m_minCorner = minc;
	m_params.m_boxEnabled = boxEnabled;
	m_data.updateParams(m_params);
    //printf("1\n");
    // get the number of occupied blocks, and save the bucket id of the occupied blocks to pass to pass 2
	extractIsoSurfacePass1CUDA(hashData, m_params, m_data);
    cudaCheckError(cudaGetLastError());
    //printf("occupied blocks: %d\n", m_data.getNumOccupiedBlocks());
    // to extract the triangles, we need to traverse the occupied blocks, and do the marching cubes on each voxel in the occupied blocks.
    //printf("2\n");
    printf("m_data.getNumOccupiedBlocks(): %d\n", m_data.getNumOccupiedBlocks());
    extractIsoSurfacePass2CUDA(hashData, m_params, m_data, m_data.getNumOccupiedBlocks());
    cudaCheckError(cudaGetLastError());
    //printf("3\n");
    //printf("In CUDAMarchingCubesHashSDF::extractIsoSurface, after extractIsoSurfacePass2CUDA\n");
}

void CUDAMarchingCubesHashSDF::export_ply(const std::string& filename)
{
    MarchingCubesData cpuData = m_data.copyToCPU();
    std::ofstream file_out { filename };
    if (!file_out.is_open())
        return;
    file_out << "ply" << std::endl;
    file_out << "format ascii 1.0" << std::endl;
    file_out << "element vertex " << 3*cpuData.d_numTriangles[0] << std::endl;
    file_out << "property float x" << std::endl;
    file_out << "property float y" << std::endl;
    file_out << "property float z" << std::endl;
    file_out << "element face " << cpuData.d_numTriangles[0] << std::endl;
    file_out << "property list uchar int vertex_index" << std::endl;
    file_out << "end_header" << std::endl;

    for (int v_idx = 0; v_idx < cpuData.d_numTriangles[0]; ++v_idx) {
        float3 v0 = cpuData.d_triangles[v_idx].v0.p;
        float3 v1 = cpuData.d_triangles[v_idx].v1.p;
        float3 v2 = cpuData.d_triangles[v_idx].v2.p;
        file_out << v0.x << " " << v0.y << " " << v0.z << " ";
        file_out << v1.x << " " << v1.y << " " << v1.z << " ";
        file_out << v2.x << " " << v2.y << " " << v2.z << " ";
    }

    for (int t_idx = 0; t_idx < 3*cpuData.d_numTriangles[0]; t_idx += 3) {
        file_out << 3 << " " << t_idx + 1 << " " << t_idx << " " << t_idx + 2 << std::endl;
    }
}