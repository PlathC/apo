#ifndef APO_CAVITY_DETECTION_SHADERS_DATA_CUH
#define APO_CAVITY_DETECTION_SHADERS_DATA_CUH

#include <cstdint>

#include <optix.h>

namespace apo
{
    struct GeometryHitGroup
    {
    };

    struct HitInfo
    {
        float t;
        int   hit = 0;

        __device__ __host__ bool hasHit() const { return static_cast<bool>( hit ); }
    };

    struct OcclusionDetectionData
    {
        OptixTraversableHandle handle;

        uint32_t       vertexNb;
        const float4 * vertices;

        uint32_t       sampleNb;
        const float3 * directions;

        float * hitRatios;
    };
} // namespace apo

#endif // APO_CAVITY_DETECTION_SHADERS_DATA_CUH