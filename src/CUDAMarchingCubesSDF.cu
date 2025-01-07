
#include <cutil_math.h>

#include "VoxelHash.h"
#include "MarchingCubesSDFUtil.h"


__global__ void resetMarchingCubesKernel(MarchingCubesData data) 
{
	*data.d_numTriangles = 0;
	*data.d_numOccupiedBlocks = 0;	
}
 
__global__ void extractIsoSurfaceKernel(HashData hashData, MarchingCubesData data) 
{
	uint idx = blockIdx.x;

	const HashEntry& entry = hashData.d_hash[idx];
	if (entry.ptr != FREE_ENTRY) {
		int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
		int3 pi = pi_base + make_int3(threadIdx);
		float3 worldPos = hashData.virtualVoxelPosToWorld(pi);

		data.extractIsoSurfaceAtPosition(worldPos, hashData);
	}
}

extern "C" void resetMarchingCubesCUDA(MarchingCubesData& data)
{
	const dim3 blockSize(1, 1, 1);
	const dim3 gridSize(1, 1, 1);

	resetMarchingCubesKernel<<<gridSize, blockSize>>>(data);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

extern "C" void extractIsoSurfaceCUDA(const HashData& hashData, const MarchingCubesParams& params, MarchingCubesData& data)
{
	const dim3 gridSize(params.m_hashNumBuckets*params.m_hashBucketSize, 1, 1);
	const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

	extractIsoSurfaceKernel<<<gridSize, blockSize>>>(hashData, data);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}
#define T_PER_BLOCK 8
__global__ void extractIsoSurfacePass1Kernel(HashData hashData, MarchingCubesData data)
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int bucketID = blockIdx.x*blockDim.x + threadIdx.x;


	if (bucketID < hashParams.m_hashNumBuckets*HASH_BUCKET_SIZE) {

		HashEntry& entry = hashData.d_hash[bucketID];

		if (entry.ptr != FREE_ENTRY) {

			uint addr = atomicAdd(&data.d_numOccupiedBlocks[0], 1);
			data.d_occupiedBlocks[addr] = bucketID;
		}
	}
}

extern "C" void extractIsoSurfacePass1CUDA(const HashData& hashData, const MarchingCubesParams& params, MarchingCubesData& data)
{
	const dim3 gridSize((params.m_hashNumBuckets*params.m_hashBucketSize + (T_PER_BLOCK*T_PER_BLOCK) - 1) / (T_PER_BLOCK*T_PER_BLOCK), 1);
	const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);
	extractIsoSurfacePass1Kernel << <gridSize, blockSize >> >(hashData, data);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}



__global__ void extractIsoSurfacePass2Kernel(HashData hashData, MarchingCubesData data)
{
	uint idx = data.d_occupiedBlocks[blockIdx.x];

	const HashEntry& entry = hashData.d_hash[idx];
	if (entry.ptr != FREE_ENTRY) {
		int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);
		int3 pi = pi_base + make_int3(threadIdx);
		float3 worldPos = hashData.virtualVoxelPosToWorld(pi);

		data.extractIsoSurfaceAtPosition(worldPos, hashData);
	}
}


extern "C" void extractIsoSurfacePass2CUDA(const HashData& hashData, const MarchingCubesParams& params, MarchingCubesData& data, unsigned int numOccupiedBlocks)
{
	const dim3 gridSize(numOccupiedBlocks, 1, 1);
	const dim3 blockSize(params.m_sdfBlockSize, params.m_sdfBlockSize, params.m_sdfBlockSize);

	if (numOccupiedBlocks) {
		extractIsoSurfacePass2Kernel << <gridSize, blockSize >> >(hashData, data);
	}
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}