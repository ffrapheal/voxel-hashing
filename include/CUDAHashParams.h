#ifndef HASHPARAMS
#define HASHPARAMS

struct HashParams {
	HashParams() {}

	unsigned int m_hashNumBuckets;
	unsigned int m_hashBucketSize;
	unsigned int m_hashMaxCollisionLinkedListSize;
	unsigned int m_numSDFBlocks;
	int m_SDFBlockSize;
	float m_virtualVoxelSize;
	unsigned int m_numOccupiedBlocks;
	float m_maxIntegrationDistance;
	float m_truncScale;
	float m_truncation;
	unsigned int m_integrationWeightSample;
	unsigned int m_integrationWeightMax;
} __attribute__((aligned(16)));

#endif