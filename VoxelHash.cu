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

//Header file

#include <VoxelHash.h>

//constant

static const int LOCK_ENTRY = -1;
static const int FREE_ENTRY = -2;
static const int NO_OFFSET = 0;
static const int UNLOCK_ENTRY = 0;

//GPU constant
__constant__ HashParams c_hashParams;

#define SDF_BLOCK_SIZE 100
#define HASH_BUCKET_SIZE 2 
#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

#ifndef PINF
#define PINF __int_as_float(0x7f800000)
#endif
	__global__
	void HashData::updatesdfframe(float3* worldpos, float3* normal)
	{
		int idx;
		insertHashEntryElement(worldpos[idx]);
		HashEntry curr = getHashEntry(worldpos[idx]);
		computesdf(worldpos[idx],normal[idx]);
		Voxel v = getVoxel(worldpos[idx]);
		combineVoxel(v,sdf);
		setVoxel(worldpos[idx],v);
	}
    __global__ 
    void HashData::initializeHashEntry(HashEntry* d_hash, int hashNumBuckets, int hashBucketSize)
    {
	    int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < hashNumBuckets * hashBucketSize) {

			d_hash[idx].pos = make_int3(0, 0, 0);
			d_hash[idx].ptr = FREE_ENTRY;                  
			d_hash[idx].offset = 0; 
		}
    }

    __global__ 
    void HashData::initializeHeap(unsigned int* d_heap, unsigned int numSDFBlocks)
    {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < numSDFBlocks) {
			d_heap[idx] = idx;
		}
    }

    __global__ 
    void HashData::initializeHeapCounter(unsigned int* d_heapCounter, unsigned int value)
    {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx == 0) {
			d_heapCounter[0] = value;
		}
    }

	__host__
	void HashData::initializeHashParams(const HashParams& params)
	{
		params.m_hashBucketSize=m_hashBucketSize;
		params.m_hashMaxCollisionLinkedListSize=m_hashMaxCollisionLinkedListSize;
		params.m_hashNumBuckets=m_hashNumBuckets;
		params.m_integrationWeightMax=m_integrationWeightMax;
		params.m_integrationWeightSample=m_integrationWeightSample;
		params.m_maxIntegrationDistance=m_maxIntegrationDistance;
		params.m_numOccupiedBlocks=m_numOccupiedBlocks;
		params.m_numSDFBlocks=m_numSDFBlocks;
		params.m_SDFBlockSize=m_SDFBlockSize;
		params.m_truncation=m_truncation;
		params.m_truncScale=m_truncScale;
		params.m_virtualVoxelSize=m_virtualVoxelSize;
	}

	__host__
	void HashData::allocate(bool dataOnGPU = true)
    {
		HashParams params;
		initializeHashParams(params);
		m_bIsOnGPU = dataOnGPU;
		if (m_bIsOnGPU) {	
			//allocate memory for heap and heapcounter.
			cudaMalloc(&d_heap, sizeof(unsigned int) * params.m_numSDFBlocks);
			cudaMalloc(&d_heapCounter, sizeof(unsigned int));
			//initialization heap and heapcounter in global function.
			int numThreads = 1024;
			int numBlocks = (params.m_numSDFBlocks + numThreads - 1) / numThreads;
			initializeHeap<<<numBlocks, numThreads>>>(d_heap, params.m_numSDFBlocks); 
			cudaDeviceSynchronize();
			initializeHeapCounter<<<1, 1>>>(d_heapCounter, params.m_numSDFBlocks-1); 
			cudaDeviceSynchronize();
			//allocate memory for hashtable.
			cudaMalloc(&d_hash, sizeof(HashEntry)* params.m_hashNumBuckets * params.m_hashBucketSize);
			//initialization heap and heapcounter in global function.
			numThreads = 1024;  
    		numBlocks = (params.m_hashNumBuckets * params.m_hashBucketSize + numThreads - 1) / numThreads;
			initializeHashEntry<<<numBlocks, numThreads>>>(d_hash, params.m_hashNumBuckets, params.m_hashBucketSize);
    		cudaDeviceSynchronize();
			//allocate memory for SDFBlocks and hashBucketMutex;
			cudaMalloc(&d_SDFBlocks, sizeof(Voxel) * params.m_numSDFBlocks * params.m_SDFBlockSize*params.m_SDFBlockSize*params.m_SDFBlockSize);
			cudaMalloc(&d_hashBucketMutex, sizeof(int)* params.m_hashNumBuckets);
			//after allocation the default value is zero so we not need to intialize.
    		}
		 else {
			//allocate memory in cpu.
			d_heap = new unsigned int[params.m_numSDFBlocks];
			d_heapCounter = new unsigned int[1];
			d_hash = new HashEntry[params.m_hashNumBuckets * params.m_hashBucketSize];
			d_SDFBlocks = new Voxel[params.m_numSDFBlocks * params.m_SDFBlockSize*params.m_SDFBlockSize*params.m_SDFBlockSize];
			d_hashBucketMutex = new int[params.m_hashNumBuckets];
		}
		//update gpu constant params.
		updateParams(params);
	}

	__host__
	void HashData::free()
    {
		//free memory 
		if (m_bIsOnGPU) {
			cudaFree(d_heap);
			cudaFree(d_heapCounter);
			cudaFree(d_hash);
			cudaFree(d_SDFBlocks);
			cudaFree(d_hashBucketMutex);
		} else {
			if (d_heap) delete[] d_heap;
			if (d_heapCounter) delete[] d_heapCounter;
			if (d_hash) delete[] d_hash;
			if (d_SDFBlocks) delete[] d_SDFBlocks;
			if (d_hashBucketMutex) delete[] d_hashBucketMutex;
		}
		d_hash = NULL;
		d_heap = NULL;
		d_heapCounter = NULL;
		d_SDFBlocks = NULL;
		d_hashBucketMutex = NULL;
    }

	__host__
	void HashData::updateParams(const HashParams& params)
    {
		if (m_bIsOnGPU) {
			size_t size;
			cudaGetSymbolSize(&size, reinterpret_cast<const void*>(&c_hashParams));
			cudaMemcpyToSymbol(reinterpret_cast<const void*>(&c_hashParams),&params, size, 0, cudaMemcpyHostToDevice);
		}
    }

	__host__
	HashData HashData::copyToCPU() const
    {
		HashParams params;
		HashData hashData;
		hashData.allocate(params, false);
		cudaMemcpy(hashData.d_heap, d_heap, sizeof(unsigned int) * params.m_numSDFBlocks, cudaMemcpyDeviceToHost);
		cudaMemcpy(hashData.d_heapCounter, d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(hashData.d_hash, d_hash, sizeof(HashEntry)* params.m_hashNumBuckets * params.m_hashBucketSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(hashData.d_SDFBlocks, d_SDFBlocks, sizeof(Voxel) * params.m_numSDFBlocks * params.m_SDFBlockSize*params.m_SDFBlockSize*params.m_SDFBlockSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(hashData.d_hashBucketMutex, d_hashBucketMutex, sizeof(int)* params.m_hashNumBuckets, cudaMemcpyDeviceToHost);		
		return hashData;
    }

	/////////////////
	// Device part //
	/////////////////
#ifdef __CUDACC__
	__device__
	float computesdf(float3 worldpos, float3 normal)
	{
		int3 voxelpos = worldToVirtualVoxelPos(worldpos);
		float3 pos1 = voxelpos*c_hashParams.m_virtualVoxelSize+1.0*c_hashParams.m_virtualVoxelSize/2-worldpos;
		return pos1 * normal;
	}

	__device__
	const HashParams& HashData::params() const
	{
		return c_hashParams;
	}

	//compute hash value to determine the hashpos
	__device__ 
	uint HashData::computeHashPos(const int3& virtualVoxelPos) const
	{
		const int p0 = 73856093;
		const int p1 = 19349669;
		const int p2 = 83492791;
		int res = ((virtualVoxelPos.x * p0) ^ (virtualVoxelPos.y * p1) ^ (virtualVoxelPos.z * p2)) % c_hashParams.m_hashNumBuckets;
		if (res < 0) res += c_hashParams.m_hashNumBuckets;
		return (uint)res;		
	}

	__device__ 
	void HashData::combineVoxel(const Voxel &v, float sdf) const
	{
		v.sdf = (v.sdf*(float)v.weight + sdf)/(v.weight+1.0);
		//out.weight = min(c_hashParams.m_integrationWeightMax, (unsigned int)v0.weight + (unsigned int)v1.weight);	
		v.weight = v.weight+1.0;	
	}

	__device__ 
	float HashData::getTruncation(float z) const
	{
		return c_hashParams.m_truncation + c_hashParams.m_truncScale * z;
	}

	//here is some pos transform.

	//voxelpos is the pos to compute the hashpos
	__device__ 
	int3 HashData::worldToVirtualVoxelPos(const float3& pos) const
	{
		const float3 p = pos / c_hashParams.m_virtualVoxelSize;
		return make_int3(p+make_float3(sign(p))*0.5f);
	}

	//to get the sdfblock pos , it is the same as the pos of the first voxel in this sdfblock within 512 voxel . 
	__device__ 
	int3 HashData::virtualVoxelPosToSDFBlock(int3 virtualVoxelPos) const
	{
		if (virtualVoxelPos.x < 0) virtualVoxelPos.x -= SDF_BLOCK_SIZE-1;
		if (virtualVoxelPos.y < 0) virtualVoxelPos.y -= SDF_BLOCK_SIZE-1;
		if (virtualVoxelPos.z < 0) virtualVoxelPos.z -= SDF_BLOCK_SIZE-1;

		return make_int3(
			virtualVoxelPos.x/SDF_BLOCK_SIZE,
			virtualVoxelPos.y/SDF_BLOCK_SIZE,
			virtualVoxelPos.z/SDF_BLOCK_SIZE);
	}

	//the reverse transform.
	__device__ 
	int3 HashData::SDFBlockToVirtualVoxelPos(const int3& sdfBlock) const
	{
		return sdfBlock*SDF_BLOCK_SIZE;
	}

	//voxel pos to world pos
	__device__ 
	float3 HashData::virtualVoxelPosToWorld(const int3& pos) const
	{
		return make_float3(pos)*c_hashParams.m_virtualVoxelSize;
	}

	// sdf pos to world pos
	__device__ 
	float3 HashData::SDFBlockToWorld(const int3& sdfBlock) const
	{
		return virtualVoxelPosToWorld(SDFBlockToVirtualVoxelPos(sdfBlock));
	}

	// world to sdfblock pos
	__device__ 
	int3 HashData::worldToSDFBlock(const float3& worldPos) const
	{
		return virtualVoxelPosToSDFBlock(worldToVirtualVoxelPos(worldPos));
	}

	//determine if it is behind the frustum.
	// __device__
	// bool isSDFBlockInCameraFrustumApprox(const int3& sdfBlock) {
	// 	float3 posWorld = virtualVoxelPosToWorld(SDFBlockToVirtualVoxelPos(sdfBlock)) + c_hashParams.m_virtualVoxelSize * 0.5f * (SDF_BLOCK_SIZE - 1.0f);
	// 	return DepthCameraData::isInCameraFrustumApprox(c_hashParams.m_rigidTransformInverse, posWorld);
	// }

	//to get the delinerize pos as voxel pos.
	__device__ 
	uint3 HashData::delinearizeVoxelIndex(uint idx) const
	{
		uint x = idx % SDF_BLOCK_SIZE;
		uint y = (idx % (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)) / SDF_BLOCK_SIZE;
		uint z = idx / (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);	
		return make_uint3(x,y,z);
	}

	//to get the linerize pos in *sdfblocks
	__device__ 
	uint HashData::linearizeVoxelPos(const int3& virtualVoxelPos) const
	{
		return  
			virtualVoxelPos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE +
			virtualVoxelPos.y * SDF_BLOCK_SIZE +
			virtualVoxelPos.x;
	}

	
	__device__ 
	int HashData::virtualVoxelPosToLocalSDFBlockIndex(const int3& virtualVoxelPos) const
	{
		int3 localVoxelPos = make_int3(
			virtualVoxelPos.x % SDF_BLOCK_SIZE,
			virtualVoxelPos.y % SDF_BLOCK_SIZE,
			virtualVoxelPos.z % SDF_BLOCK_SIZE);

		if (localVoxelPos.x < 0) localVoxelPos.x += SDF_BLOCK_SIZE;
		if (localVoxelPos.y < 0) localVoxelPos.y += SDF_BLOCK_SIZE;
		if (localVoxelPos.z < 0) localVoxelPos.z += SDF_BLOCK_SIZE;

		return linearizeVoxelPos(localVoxelPos);
	}

	__device__ 
	int HashData::worldToLocalSDFBlockIndex(const float3& world) const
	{
		int3 virtualVoxelPos = worldToVirtualVoxelPos(world);
		return virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos);		
	}

	__device__ 
	HashEntry HashData::getHashEntry(const float3& worldPos) const
	{
		int3 blockID = worldToSDFBlock(worldPos);
		return getHashEntryForSDFBlockPos(blockID);
	}

	// __device__
	// void HashData::deleteHashEntry(const float3& worldPos) const
	// {
	// 	uint hashpos = HashData::computeHashPos(HashData::)
	// }

	__device__ 
	void HashData::deleteHashEntry(uint id)
	{
		deleteHashEntry(d_hash[id]);
	}

	__device__ 
	void HashData::deleteHashEntry(HashEntry& hashEntry)
	{
		hashEntry.pos = make_int3(0);
		hashEntry.offset = 0;
		hashEntry.ptr = FREE_ENTRY;
	}

	__device__ 
	bool HashData::voxelExists(const float3& worldPos) const
	{
		HashEntry hashEntry = getHashEntry(worldPos);
		return (hashEntry.ptr != FREE_ENTRY);
	}

	__device__  
	void HashData::deleteVoxel(Voxel& v) const
	{
		v.color = make_uchar3(0,0,0);
		v.weight = 0;
		v.sdf = 0.0f;
	}
    
	__device__ 
	void HashData::deleteVoxel(uint id)
	{
		deleteVoxel(d_SDFBlocks[id]);
	}

	__device__ 
	Voxel HashData::getVoxel(const float3& worldPos) const
	{
		HashEntry hashEntry = getHashEntry(worldPos);
		Voxel v;
		if (hashEntry.ptr == FREE_ENTRY) {
			deleteVoxel(v);			
		} else {
			int3 virtualVoxelPos = worldToVirtualVoxelPos(worldPos);
			v = d_SDFBlocks[hashEntry.ptr + virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos)];
		}
		return v;
	}

	__device__ 
	Voxel HashData::getVoxel(const int3& virtualVoxelPos) const
	{
		HashEntry hashEntry = getHashEntryForSDFBlockPos(virtualVoxelPosToSDFBlock(virtualVoxelPos));
		Voxel v;
		if (hashEntry.ptr == FREE_ENTRY) {
			deleteVoxel(v);			
		} else {
			v = d_SDFBlocks[hashEntry.ptr + virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos)];
		}
		return v;
	}
	
	__device__ 
	int HashData::setVoxel(const int3& virtualVoxelPos, Voxel& voxelInput) const
	{
		HashEntry hashEntry = getHashEntryForSDFBlockPos(virtualVoxelPosToSDFBlock(virtualVoxelPos));
		return hashEntry.ptr;
		if (hashEntry.ptr != FREE_ENTRY) {
			d_SDFBlocks[hashEntry.ptr + virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos)] = voxelInput;
			return true;
		}
		return false;
	}

	__device__ 
	int HashData::setVoxel(const float3& worldPos, Voxel& voxelInput) const
	{
		int3 voxelpos = worldToVirtualVoxelPos(worldpos);
		HashEntry hashEntry = getHashEntryForSDFBlockPos(virtualVoxelPosToSDFBlock(voxelpos));
		return hashEntry.ptr;
		if (hashEntry.ptr != FREE_ENTRY) {
			d_SDFBlocks[hashEntry.ptr + virtualVoxelPosToLocalSDFBlockIndex(voxelpos)] = voxelInput;
			return true;
		}
		return false;
	}

	__device__ 
	HashEntry HashData::getHashEntryForSDFBlockPos(const int3& sdfBlock) const
	{
		uint h = computeHashPos(sdfBlock);			//hash bucket
		uint hp = h * HASH_BUCKET_SIZE;	//hash position

		HashEntry entry;
		entry.pos = sdfBlock;
		entry.offset = 0;
		entry.ptr = FREE_ENTRY;

		for (uint j = 0; j < HASH_BUCKET_SIZE; j++) {
			uint i = j + hp;
			HashEntry curr = d_hash[i];
			if (curr.pos.x == entry.pos.x && curr.pos.y == entry.pos.y && curr.pos.z == entry.pos.z && curr.ptr != FREE_ENTRY) {
				return curr;
			}
		}
		return entry;
	}

	__device__ 
	unsigned int HashData::getNumHashEntriesPerBucket(unsigned int bucketID)
	{
		unsigned int h = 0;
		for (uint i = 0; i < HASH_BUCKET_SIZE; i++) {
			if (d_hash[bucketID*HASH_BUCKET_SIZE+i].ptr != FREE_ENTRY) {
				h++;
			}
		} 
		return h;
	}

	__device__ 
	unsigned int HashData::getNumHashLinkedList(unsigned int bucketID)
	{
		unsigned int listLen = 0;		
		return listLen;
	}

	__device__
	uint HashData::consumeHeap()
	{
		uint addr = atomicSub(&d_heapCounter[0], 1);
		//TODO MATTHIAS check some error handling?
		return d_heap[addr];
	}

	__device__
	void HashData::appendHeap(uint ptr)
	{
		uint addr = atomicAdd(&d_heapCounter[0], 1);
		//TODO MATTHIAS check some error handling?
		d_heap[addr+1] = ptr;
	}

    __device__
	void HashData::insertHashEntryElement(const float3& worldpos)
	{
		int3 pos = worldToVirtualVoxelPos(worldpos);
		uint h = computeHashPos(pos);				//hash bucket
		uint hp = h * HASH_BUCKET_SIZE;	//hash position

		int firstEmpty = -1;
		for (uint j = 0; j < HASH_BUCKET_SIZE; j++) {
			uint i = j + hp;		
			const HashEntry& curr = d_hash[i];

			//in that case the SDF-block is already allocated and corresponds to the current position -> exit thread
			if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
				return;
			}

			//store the first FREE_ENTRY hash entry
			if (firstEmpty == -1 && curr.ptr == FREE_ENTRY) {
				firstEmpty = i;
			}
		}


		if (firstEmpty != -1) {	//if there is an empty entry and we haven't allocated the current entry before
			//int prevValue = 0;
			//InterlockedExchange(d_hashBucketMutex[h], LOCK_ENTRY, prevValue);	//lock the hash bucket
			int prevValue = atomicExch(&d_hashBucketMutex[h], LOCK_ENTRY);
			if (prevValue != LOCK_ENTRY) {	//only proceed if the bucket has been locked
				HashEntry& entry = d_hash[firstEmpty];
				entry.pos = pos;
				entry.offset = NO_OFFSET;		
				entry.ptr = consumeHeap() * SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;	//memory alloc
			}
			return;
		}
	}

	__device__
	bool HashData::deleteHashEntryElement(const int3& sdfBlock)
	{

	}

#endif