#ifndef VOXEL_HASH
#define VOXEL_HASH

//Header file
#include <cuda_runtime.h>
#include "CUDAHashParams.h"
#include <stddef.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <cutil_math.h>
#define SDF_BLOCK_SIZE 100
//rename data

#ifndef sint
typedef signed int sint;
#endif

#ifndef uint
typedef unsigned int uint;
#endif 

#ifndef slong 
typedef signed long slong;
#endif

#ifndef ulong
typedef unsigned long ulong;
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef schar
typedef signed char schar;
#endif

//constant

static const int LOCK_ENTRY = -1;
static const int FREE_ENTRY = -2;
static const int NO_OFFSET = 0;
static const int UNLOCK_ENTRY = 0;

//struct define
struct HashEntry 
{
	int3	pos;		//hash position (lower left corner of SDFBlock))
	int		ptr;		//pointer into heap to SDFBlock
	uint	offset;		//offset for collisions

	
	__device__ void operator=(const struct HashEntry& e) {
		//((int*)this)[0] = ((const int*)&e)[0];
		//((int*)this)[1] = ((const int*)&e)[1];
		//((int*)this)[2] = ((const int*)&e)[2];
		//((int*)this)[3] = ((const int*)&e)[3];
		//((int*)this)[4] = ((const int*)&e)[4];
		((long long*)this)[0] = ((const long long*)&e)[0];
		((long long*)this)[1] = ((const long long*)&e)[1];
		((int*)this)[4] = ((const int*)&e)[4];
	}
} __attribute__((aligned(16)));

struct Voxel {
	float	sdf;		//signed distance function
	//uchar3	color;		//color 
	uchar	weight;		//accumulated sdf weight

	__device__ void operator=(const struct Voxel& v) {
		//((int*)this)[0] = ((const int*)&v)[0];
		//((int*)this)[1] = ((const int*)&v)[1];
		((long long*)this)[0] = ((const long long*)&v)[0];
	}

}__attribute__((aligned(8)));
__global__
void updatesdfframe(float3* pos, float3* normal);

__global__ 
void initializeHashEntry(HashEntry* d_hash, int hashNumBuckets, int hashBucketSize);

__global__ 
void initializeHeap(unsigned int* d_heap, unsigned int numSDFBlocks);

__global__ 
void initializeHeapCounter(unsigned int* d_heapCounter, unsigned int value);
class HashData {

	///////////////
	// Host part //
	///////////////

	public:
    __device__ __host__
	HashData() {
		d_heap = NULL;
		d_heapCounter = NULL;
		d_hash = NULL;
		// d_hashDecision = NULL;
		// d_hashDecisionPrefix = NULL;
		// d_hashCompactified = NULL;
		// d_hashCompactifiedCounter = NULL;
		d_SDFBlocks = NULL;
		d_hashBucketMutex = NULL;
		m_bIsOnGPU = false;
	}

	__host__
	void allocate(bool dataOnGPU = true);

	__host__
	void free();

	__host__
	void updateParams(const HashParams& params);

	__host__
	HashData copyToCPU() const;
	
	__host__
	void initializeHashParams(HashParams& params);
	/////////////////
	// Device part //
	/////////////////
#ifdef __CUDACC__
	__device__
	float computesdf(float3 worldpos, float3 normal);

	__device__
	const HashParams& params() const;

	__device__ 
	uint computeHashPos(const float3& WorldPos) const;

	__device__ 
	void combineVoxel(Voxel &v, float sdf) const;

	__device__ 
	float getTruncation(float z) const;

	__device__ 
	int3 worldToVirtualVoxelPos(const float3& pos) const;

	__device__ 
	int3 virtualVoxelPosToSDFBlock(int3 virtualVoxelPos) const;

	__device__ 
	int3 SDFBlockToVirtualVoxelPos(const int3& sdfBlock) const;

	__device__ 
	float3 virtualVoxelPosToWorld(const int3& pos) const;

	__device__ 
	float3 SDFBlockToWorld(const int3& sdfBlock) const;

	__device__ 
	int3 worldToSDFBlock(const float3& worldPos) const;

	//determine if it is behind the frustum.
	// __device__
	// bool isSDFBlockInCameraFrustumApprox(const int3& sdfBlock) {
	// 	float3 posWorld = virtualVoxelPosToWorld(SDFBlockToVirtualVoxelPos(sdfBlock)) + c_hashParams.m_virtualVoxelSize * 0.5f * (SDF_BLOCK_SIZE - 1.0f);
	// 	return DepthCameraData::isInCameraFrustumApprox(c_hashParams.m_rigidTransformInverse, posWorld);
	// }

	__device__ 
	uint3 delinearizeVoxelIndex(uint idx) const;

	__device__ 
	uint linearizeVoxelPos(const int3& virtualVoxelPos)	const;

	__device__ 
	int WorldPosToLocalSDFBlockIndex(const float3& WorldPos) const;

	__device__ 
	HashEntry getHashEntry(const float3& WorldPos) const;


	__device__ 
	void deleteHashEntry(uint id);

	__device__ 
	void deleteHashEntry(HashEntry& hashEntry);

	__device__ 
	bool voxelExists(const float3& worldPos) const;

	__device__  
	void deleteVoxel(Voxel& v) const;

	__device__ 
	void deleteVoxel(uint id);


	__device__ 
	Voxel getVoxel(const float3& worldPos) const;

	__device__ 
	int setVoxel(const float3& worldPos, Voxel& voxelInput) const;

	__device__ 
	HashEntry getHashEntryForWorldPos(const float3& WorldPos) const;

	__device__ 
	unsigned int getNumHashEntriesPerBucket(unsigned int bucketID);

	__device__ 
	unsigned int getNumHashLinkedList(unsigned int bucketID);

	__device__
	uint consumeHeap();

	__device__
	void appendHeap(uint ptr);

    __device__
	void insertHashEntryElement(const float3& worldpos);

	__device__
	bool deleteHashEntryElement(const int3& sdfBlock);

#endif	//CUDACC

	uint*		d_heap;						//heap that manages free memory
	uint*		d_heapCounter;				//single element; used as an atomic counter (points to the next free block)
	HashEntry*	d_hash;						//hash that stores pointers to sdf blocks
	Voxel*		d_SDFBlocks;				//sub-blocks that contain 8x8x8 voxels (linearized); are allocated by heap
	int*		d_hashBucketMutex;			//binary flag per hash bucket; used for allocation to atomically lock a bucket

	bool		m_bIsOnGPU;					//the class be be used on both cpu and gpu
};
#endif