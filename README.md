# voxel-hashing
The `Voxelhash.h` is the main code file.
# introduce
`void allocate(const HashParams& params, bool dataOnGPU = true)`  
分配内存，决定位置  
`void free()`  
与allocate对应  
`HashData copyToCPU() const`  
复制一个新的hashdata到cpu中查看  
`uint computeHashPos(const int3& virtualVoxelPos) const`  
通过体素位置判断在hash表中的位置。注意区分体素和体素块位置区别  
`void combineVoxel(const Voxel &v0, const Voxel& v1, Voxel &out) const`  
多视角下同一体素融合  
`float getTruncation(float z) const `    
返回sdf截断距离  
`float3 worldToVirtualVoxelPosFloat(const float3& pos) const	 `  
将世界坐标转为体素坐标，浮点数  
`int3 worldToVirtualVoxelPos(const float3& pos) const `  
将世界坐标转为体素坐标整数，通常采用这个  
`int3 virtualVoxelPosToSDFBlock(int3 virtualVoxelPos) const `  
将体素坐标转为体素块坐标，512个体素对应同一个体素块坐标  
`int3 SDFBlockToVirtualVoxelPos(const int3& sdfBlock) const	`  
将体素块坐标转为体素坐标，1对1  
`float3 virtualVoxelPosToWorld(const int3& pos) const	`    
将体素坐标转为世界坐标，1对1  
`float3 SDFBlockToWorld(const int3& sdfBlock) const	`  
将体素块坐标转换为世界坐标，1对1  
`int3 worldToSDFBlock(const float3& worldPos) const`    
将世界坐标转换为体素块坐标  
`uint3 delinearizeVoxelIndex(uint idx) const`  
根据序号（即体素块线性化后512之一），获取当前体素的在体素块的局部坐标  
`uint linearizeVoxelPos(const int3& virtualVoxelPos)	const`  
根据体素局部坐标，获取在该体素块的序号  
`int virtualVoxelPosToLocalSDFBlockIndex(const int3& virtualVoxelPos) const`  
根据体素全局坐标获取体素在体素块局部坐标  
`int worldToLocalSDFBlockIndex(const float3& world) const`  
从世界坐标获取体素在体素块局部坐标  
`HashEntry getHashEntry(const float3& worldPos) const	`  
根据世界坐标获取hash表中的条目，请注意这里获取的是对应体素块的坐标，这意味着512个相邻体素会对应到同一个哈希条目  
`void deleteHashEntry(uint id) `  `void deleteHashEntry(HashEntry& hashEntry)`  
根据在hash表中的序号，删除hash表中的数据，实际上是置为0，因为内存在一开始已经分配完毕  
`bool voxelExists(const float3& worldPos) const	`  
判断该世界坐标对应体素是否存在  
`void deleteVoxel(Voxel& v) const `  `void deleteVoxel(uint id)`  
根据序号，这个序号得到的方法是根据体素所在的全局序号，删除体素，实际置0  
`Voxel getVoxel(const float3& worldPos) const`  
由世界坐标获取体素  
`Voxel getVoxel(const int3& virtualVoxelPos) const`  
由体素坐标获取体素  
`void setVoxel(const int3& virtualVoxelPos, Voxel& voxelInput) const`  
由虚拟体素坐标和一个体素对象将体素复制到哈希表中  
`HashEntry getHashEntryForSDFBlockPos(const int3& sdfBlock) const`  
由体素块坐标获取哈希条目，如果不存在会创建条目  
`unsigned int getNumHashEntriesPerBucket(unsigned int bucketID)`  
判断某个哈希桶有多少个哈希条目，哈希桶中存放着相同的哈希值的哈希条目  
`unsigned int getNumHashLinkedList(unsigned int bucketID)`  
判断某个哈希桶链表有多少哈希条目  
`uint consumeHeap()`  `void appendHeap(uint ptr)`  
堆中分配内存，原子操作，堆中主要存放哈希表哈希条目  
`void allocBlock(const int3& pos)`  
根据体素块坐标给某一个体素块分配哈希条目  
`bool insertHashEntry(HashEntry entry)`  
插入某个哈希条目并判断是否插入成功  
`bool deleteHashEntryElement(const int3& sdfBlock)`  
根据体素块位置删除哈希条目  
