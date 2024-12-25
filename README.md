# voxel-hashing

The `Voxelhash.h` is the main code file.

## function interface

`void allocate(const HashParams& params, bool dataOnGPU = true)`  
allocte the space, and decide the specific slots in the hash table

`void free()`  
free memory acquired by allocate 

`HashData copyToCPU() const`  
helper function, copy a new hash data to cpu

`uint computeHashPos(const int3& virtualVoxelPos) const`  
use the position of voxel(world coordinate) to compute the specific slots in hash table

`void combineVoxel(const Voxel &v0, const Voxel& v1, Voxel &out) const`  
combine the voxel from multiple view points  

`float getTruncation(float z) const`    
return the value of sdf cut-off

`float3 worldToVirtualVoxelPosFloat(const float3& pos) const`  
change world coordinate into voxel position

`int3 worldToVirtualVoxelPos(const float3& pos) const`  
change the world coordinate into voxel coordinate by integer, usually used  

`int3 virtualVoxelPosToSDFBlock(int3 virtualVoxelPos) const`  
change voxel coordinate into voxel block's coordinate, 512 voxel construct a same voxel block

`int3 SDFBlockToVirtualVoxelPos(const int3& sdfBlock) const`  
change voxel block coordinate into voxel coordinate, one-to-one mapping

`float3 virtualVoxelPosToWorld(const int3& pos) const`    
change voxel coordinate into world coordinate, one-to-one mapping

`float3 SDFBlockToWorld(const int3& sdfBlock) const`  
change voxel block coordinate into world coordinate, one-to-one mapping

`int3 worldToSDFBlock(const float3& worldPos) const`    
swithc world coordinate into voxel block coordinate

`uint3 delinearizeVoxelIndex(uint idx) const`  
use index of voxels inside serialized voxel block, to acquire current voxel's local coordinate inside voxel block

`uint linearizeVoxelPos(const int3& virtualVoxelPos) const`  
use voxel's local coordinate to acquire the offset inside the voxel block

`int virtualVoxelPosToLocalSDFBlockIndex(const int3& virtualVoxelPos) const`  
use voxel's global coordinate to acquire voxel's local coordinate inside the voxel block  

`int worldToLocalSDFBlockIndex(const float3& world) const`  
from world coordinate to acquire voxel's local index inside voxel block

`HashEntry getHashEntry(const float3& worldPos) const`  
get hash slot according to the world coordinate, the entry stores the coordinate of corresponding voxel block, 512 neighboring voxel will be mapped into the same hash entry  

`void deleteHashEntry(uint id)`  
`void deleteHashEntry(HashEntry& hashEntry)`  
delete hash entry according to the id inside hash table. But it is a lazy deletion, just clear the data to 0, enable it to be allocated to other voxel again. The memory won't change because the memory is allocated at the beginning.

`bool voxelExists(const float3& worldPos) const`  
judge whether the worldPos's voxel exists.  

`void deleteVoxel(Voxel& v) const`  
`void deleteVoxel(uint id)`  
global id of voxel, use this to delete the voxel, do lazy deletion(set it to 0)

`Voxel getVoxel(const float3& worldPos) const`  
get voxel according to world position

`Voxel getVoxel(const int3& virtualVoxelPos) const`  
get voxel according to local position

`void setVoxel(const int3& virtualVoxelPos, Voxel& voxelInput) const`  
copy a voxel into the hash entry using local position and voxel value input 

`HashEntry getHashEntryForSDFBlockPos(const int3& sdfBlock) const`  
acquire hash entry by using voxel block coordinate, if not exists, create the entry and then return

`unsigned int getNumHashEntriesPerBucket(unsigned int bucketID)`  
get the number of hash entry given a hash bucket id, a bucket hold the hash entries with the same hash value

`unsigned int getNumHashLinkedList(unsigned int bucketID)`  
find the number of linkedlist inside the bucket  

`uint consumeHeap()`  
`void appendHeap(uint ptr)`  
allocate memory in heap, which is atomic, to store hash table and hash entries inside the heap

`void allocBlock(const int3& pos)`  
allocate hash entries to a voxel block coordinate

`bool insertHashEntry(HashEntry entry)`  
insert some hash entry inside the hash table, and give the result of operation  

`bool deleteHashEntryElement(const int3& sdfBlock)`  
delete hash entry according to voxel block's coordinate  
