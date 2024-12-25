# voxel-hashing

The `Voxelhash.h` is the main code file.

## function interface

`void allocate(const HashParams& params, bool dataOnGPU = true)`  
*Usage: allocte the space, and decide the specific slots in the hash table*

`void free()`  
*Usage: free memory acquired by allocate* 

`HashData copyToCPU() const`  
*Usage: helper function, copy a new hash data to cpu*

`uint computeHashPos(const int3& virtualVoxelPos) const`  
*Usage: use the position of voxel(world coordinate) to compute the specific slots in hash table*

`void combineVoxel(const Voxel &v0, const Voxel& v1, Voxel &out) const`  
*Usage: combine the voxel from multiple view points*

`float getTruncation(float z) const`    
*Usage: return the value of sdf cut-off*

`float3 worldToVirtualVoxelPosFloat(const float3& pos) const`  
*Usage: change world coordinate into voxel position*

`int3 worldToVirtualVoxelPos(const float3& pos) const`  
*Usage: change the world coordinate into voxel coordinate by integer, usually used*  

`int3 virtualVoxelPosToSDFBlock(int3 virtualVoxelPos) const`  
*Usage: change voxel coordinate into voxel block's coordinate, 512 voxel construct a same voxel block*

`int3 SDFBlockToVirtualVoxelPos(const int3& sdfBlock) const`  
*Usage: change voxel block coordinate into voxel coordinate, one-to-one mapping*

`float3 virtualVoxelPosToWorld(const int3& pos) const`    
*Usage: change voxel coordinate into world coordinate, one-to-one mapping*

`float3 SDFBlockToWorld(const int3& sdfBlock) const`  
*Usage: change voxel block coordinate into world coordinate, one-to-one mapping*

`int3 worldToSDFBlock(const float3& worldPos) const`    
*Usage: switch world coordinate into voxel block coordinate*

`uint3 delinearizeVoxelIndex(uint idx) const`  
*Usage: use index of voxels inside serialized voxel block, to acquire current voxel's local coordinate inside voxel block*

`uint linearizeVoxelPos(const int3& virtualVoxelPos) const`  
*Usage: use voxel's local coordinate to acquire the offset inside the voxel block*

`int virtualVoxelPosToLocalSDFBlockIndex(const int3& virtualVoxelPos) const`  
*Usage: use voxel's global coordinate to acquire voxel's local coordinate inside the voxel block*  

`int worldToLocalSDFBlockIndex(const float3& world) const`  
*Usage: from world coordinate to acquire voxel's local index inside voxel block*

`HashEntry getHashEntry(const float3& worldPos) const`  
*Usage: get hash slot according to the world coordinate, the entry stores the coordinate of corresponding voxel block, 512 neighboring voxel will be mapped into the same hash entry*  

`void deleteHashEntry(uint id)`  
`void deleteHashEntry(HashEntry& hashEntry)`  
*Usage: delete hash entry according to the id inside hash table. But it is a lazy deletion, just clear the data to 0, enable it to be allocated to other voxel again. The memory won't change because the memory is allocated at the beginning.*

`bool voxelExists(const float3& worldPos) const`  
*Usage: judge whether the worldPos's voxel exists.*  

`void deleteVoxel(Voxel& v) const`  
`void deleteVoxel(uint id)`  
*Usage: global id of voxel, use this to delete the voxel, do lazy deletion(set it to 0)*

`Voxel getVoxel(const float3& worldPos) const`  
*Usage: get voxel according to world position*

`Voxel getVoxel(const int3& virtualVoxelPos) const`  
*Usage: get voxel according to local position*

`void setVoxel(const int3& virtualVoxelPos, Voxel& voxelInput) const`  
*Usage: copy a voxel into the hash entry using local position and voxel value input* 

`HashEntry getHashEntryForSDFBlockPos(const int3& sdfBlock) const`  
*Usage: acquire hash entry by using voxel block coordinate, if not exists, create the entry and then return*

`unsigned int getNumHashEntriesPerBucket(unsigned int bucketID)`  
*Usage: get the number of hash entry given a hash bucket id, a bucket hold the hash entries with the same hash value*

`unsigned int getNumHashLinkedList(unsigned int bucketID)`  
*Usage: find the number of linkedlist inside the bucket*  

`uint consumeHeap()`  
`void appendHeap(uint ptr)`  
*Usage: allocate memory in heap, which is atomic, to store hash table and hash entries inside the heap*

`void allocBlock(const int3& pos)`  
*Usage: allocate hash entries to a voxel block coordinate*

`bool insertHashEntry(HashEntry entry)`  
*Usage: insert some hash entry inside the hash table, and give the result of operation*

`bool deleteHashEntryElement(const int3& sdfBlock)`  
*Usage: delete hash entry according to voxel block's coordinate*  
