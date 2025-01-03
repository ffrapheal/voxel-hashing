#ifndef HASHCONSTANT
#define HASHCONSTANT
	unsigned int	m_hashNumBuckets=100;
	unsigned int	m_hashBucketSize=100;
	unsigned int	m_hashMaxCollisionLinkedListSize=100;
	unsigned int	m_numSDFBlocks=100;
	int				m_SDFBlockSize=100;
	float			m_virtualVoxelSize=0.5;
	unsigned int	m_numOccupiedBlocks=0;
	float			m_maxIntegrationDistance=20;
	float			m_truncScale=5;
	float			m_truncation=5;
	unsigned int	m_integrationWeightSample=5;
	unsigned int	m_integrationWeightMax=5;
#endif