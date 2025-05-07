#include <helper_math.h>
#include <optix.h>

#include "data.cuh"

using namespace apo;

extern "C"
{
    __constant__ OcclusionDetectionData parameters;
}

template<class Type>
__host__ __device__ uint2 u64ToU32x2( Type * ptr )
{
    const uint64_t uPtr = reinterpret_cast<uint64_t>( ptr );
    return make_uint2( uPtr >> 32, uPtr & 0x00000000ffffffff );
}

template<class Type>
__host__ __device__ Type * u32x2ToType( uint2 packed )
{
    return reinterpret_cast<Type *>( static_cast<uint64_t>( packed.x ) << 32 | packed.y );
}

extern "C" __global__ void __raygen__rg()
{
    const uint3    idx      = optixGetLaunchIndex();
    const uint32_t vertexId = idx.x;
    if ( vertexId >= parameters.vertexNb )
        return;

    parameters.hitRatios[ vertexId ] = 0.f;

    const float4 vertex = parameters.vertices[ vertexId ];

    HitInfo      shadowHitInfo {};
    uint2        shadowPrd = u64ToU32x2( &shadowHitInfo );
    const float3 ro        = make_float3( vertex );

    uint32_t hitNb = 0;
    for ( uint32_t s = 0; s < parameters.sampleNb; s++ )
    {
        const float3 rd = normalize( parameters.directions[ s ] );

        optixTrace( parameters.handle,
                    ro,
                    rd,
                    0.01f, // tmin
                    1e16f, // tmax
                    0.0f,  // rayTime
                    OptixVisibilityMask( 1 ),
                    OPTIX_RAY_FLAG_NONE,
                    0, // SBT offset
                    1, // SBT stride
                    0, // missSBTIndex
                    shadowPrd.x,
                    shadowPrd.y );

        if ( shadowHitInfo.hasHit() )
        {
            hitNb++;
            if ( shadowHitInfo.t < vertex.w - 1e-2f )
                printf( "Problem : t (%.2f) < w (%.2f)\n", shadowHitInfo.t, vertex.w );
        }
    }

    parameters.hitRatios[ vertexId ] = static_cast<float>( hitNb ) / static_cast<float>( parameters.sampleNb );
}

extern "C" __global__ void __miss__general()
{
    HitInfo * hitInfo = u32x2ToType<HitInfo>( make_uint2( optixGetPayload_0(), optixGetPayload_1() ) );
    hitInfo->hit      = false;
}

extern "C" __global__ void __closesthit__site()
{
    HitInfo * hitInfo = u32x2ToType<HitInfo>( make_uint2( optixGetPayload_0(), optixGetPayload_1() ) );
    hitInfo->t        = optixGetRayTmax();
    hitInfo->hit      = true;
}
