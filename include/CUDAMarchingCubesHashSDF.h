#ifndef CUDAMARCHINGCUBESHASHSDF_H
#define CUDAMARCHINGCUBESHASHSDF_H

#include "VoxelHash.h"
#include "MarchingCubesSDFUtil.h"
struct vec3f {
    float x, y, z;

    vec3f() : x(0), y(0), z(0) {}
    vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
};
class CUDAMarchingCubesHashSDF
{
public:
	CUDAMarchingCubesHashSDF(const MarchingCubesParams& params) {
		create(params);
	}
	~CUDAMarchingCubesHashSDF(void) {
		destroy();
	}

	static MarchingCubesParams parametersFromGlobalAppState(int s_marchingCubesMaxNumTriangles, int s_SDFMarchingCubeThreshFactor, int s_SDFVoxelSize,int s_hashNumBuckets) {
		MarchingCubesParams params;
		params.m_maxNumTriangles = s_marchingCubesMaxNumTriangles;
		params.m_threshMarchingCubes = s_SDFMarchingCubeThreshFactor*s_SDFVoxelSize;
		params.m_threshMarchingCubes2 = s_SDFMarchingCubeThreshFactor*s_SDFVoxelSize;
		//params.m_sdfBlockSize = SDF_BLOCK_SIZE;
		//params.m_hashBucketSize = HASH_BUCKET_SIZE;
		params.m_hashNumBuckets = s_hashNumBuckets;
		return params;
	}

	void extractIsoSurface(const HashData& hashData, const HashParams& hashParams, const vec3f& minCorner = vec3f(0.0f, 0.0f, 0.0f), const vec3f& maxCorner = vec3f(0.0f, 0.0f, 0.0f), bool boxEnabled = false);
	void export_ply(const std::string& filename);
private:
	void create(const MarchingCubesParams& params);
	void destroy(void);
	MarchingCubesParams m_params;
	MarchingCubesData	m_data;
};

#endif // CUDAMARCHINGCUBESHASHSDF_H