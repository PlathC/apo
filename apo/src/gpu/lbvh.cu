#include <thrust/sort.h>

#include "apo/gpu/lbvh.cuh"

// Strongly based on "Performance Comparison of Bounding Volume Hierarchies for GPU Ray Tracing"'s
// implementation of LBVH by Daniel Meister and Jirı Bittner
// https://github.com/meistdan/hippie/blob/main/src/hippie/rt/bvh/LBVHBuilderKernels.cu

namespace apo::gpu
{
    __device__ uint32_t mortonCode( uint32_t x, uint32_t y, uint32_t z )
    {
        x = ( x | ( x << 16 ) ) & 0x030000FF;
        x = ( x | ( x << 8 ) ) & 0x0300F00F;
        x = ( x | ( x << 4 ) ) & 0x030C30C3;
        x = ( x | ( x << 2 ) ) & 0x09249249;
        y = ( y | ( y << 16 ) ) & 0x030000FF;
        y = ( y | ( y << 8 ) ) & 0x0300F00F;
        y = ( y | ( y << 4 ) ) & 0x030C30C3;
        y = ( y | ( y << 2 ) ) & 0x09249249;
        z = ( z | ( z << 16 ) ) & 0x030000FF;
        z = ( z | ( z << 8 ) ) & 0x0300F00F;
        z = ( z | ( z << 4 ) ) & 0x030C30C3;
        z = ( z | ( z << 2 ) ) & 0x09249249;
        return x | ( y << 1 ) | ( z << 2 );
    }

    __device__ uint64_t mortonCode64( uint32_t x, uint32_t y, uint32_t z )
    {
        unsigned int       loX = x & 1023u;
        unsigned int       loY = y & 1023u;
        unsigned int       loZ = z & 1023u;
        unsigned int       hiX = x >> 10u;
        unsigned int       hiY = y >> 10u;
        unsigned int       hiZ = z >> 10u;
        unsigned long long lo  = mortonCode( loX, loY, loZ );
        unsigned long long hi  = mortonCode( hiX, hiY, hiZ );
        return ( hi << 30 ) | lo;
    }

    __device__ uint64_t mortonCode64( const float3 & centroid )
    {
        unsigned int scale = ( 1u << 20 ) - 1;
        unsigned int x     = centroid.x * scale;
        unsigned int y     = centroid.y * scale;
        unsigned int z     = centroid.z * scale;
        return mortonCode64( x, y, z );
    }

    __global__ void getMortonCodes( const uint32_t      count,
                                    const float3        sceneMin,
                                    const float3        scale,
                                    const Real4 * const spheres,
                                    uint32_t * const    indices,
                                    uint64_t * const    mortonCodes )
    {
        const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i >= count )
            return;

        indices[ i ] = i;

        const Real4    sphere   = spheres[ i ];
        const float3   centroid = make_float3( toFloat( sphere ) );
        const uint64_t code     = mortonCode64( ( centroid - sceneMin ) * scale );
        mortonCodes[ i ]        = code;
    }

    __global__ void getMortonCodesf( const uint32_t       count,
                                     const float3         sceneMin,
                                     const float3         scale,
                                     const float4 * const spheres,
                                     uint32_t * const     indices,
                                     uint64_t * const     mortonCodes )
    {
        const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i >= count )
            return;

        indices[ i ] = i;

        const float4   sphere   = spheres[ i ];
        const float3   centroid = make_float3( sphere );
        const uint64_t code     = mortonCode64( ( centroid - sceneMin ) * scale );
        mortonCodes[ i ]        = code;
    }

    __global__ void setLeaves( const uint32_t         count,
                               const Real4 *          spheres,
                               const uint32_t * const indices,
                               int *                  nodeLeftIndices,
                               int *                  nodeRightIndices,
                               float4 *               nodesAabbs,
                               Real4 *                sortedSpheres )
    {
        const uint32_t boxIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if ( boxIndex >= count )
            return;

        const int nodeIndex           = static_cast<int>( boxIndex ) + count - 1;
        nodeLeftIndices[ nodeIndex ]  = static_cast<int>( boxIndex );
        nodeRightIndices[ nodeIndex ] = static_cast<int>( boxIndex ) + 1;

        const uint32_t index = indices[ boxIndex ];

        const Real4  sphere = spheres[ index ];
        const float4 boxMin = make_float4( sphere.x - sphere.w, sphere.y - sphere.w, sphere.z - sphere.w, 0.f );
        const float4 boxMax = make_float4( sphere.x + sphere.w, sphere.y + sphere.w, sphere.z + sphere.w, 0.f );

        nodesAabbs[ nodeIndex * 2 + 0 ] = boxMin;
        nodesAabbs[ nodeIndex * 2 + 1 ] = boxMax;

        sortedSpheres[ boxIndex ] = sphere;
    }

    __global__ void setLeavesf( const uint32_t         count,
                                const float4 *         spheres,
                                const uint32_t * const indices,
                                int *                  nodeLeftIndices,
                                int *                  nodeRightIndices,
                                float4 *               nodesAabbs,
                                float4 *               sortedSpheres )
    {
        const uint32_t boxIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if ( boxIndex >= count )
            return;

        const int nodeIndex           = static_cast<int>( boxIndex ) + count - 1;
        nodeLeftIndices[ nodeIndex ]  = static_cast<int>( boxIndex );
        nodeRightIndices[ nodeIndex ] = static_cast<int>( boxIndex ) + 1;

        const uint32_t index = indices[ boxIndex ];

        const float4 sphere = spheres[ index ];
        const float4 boxMin = make_float4( sphere.x - sphere.w, sphere.y - sphere.w, sphere.z - sphere.w, 0.f );
        const float4 boxMax = make_float4( sphere.x + sphere.w, sphere.y + sphere.w, sphere.z + sphere.w, 0.f );

        nodesAabbs[ nodeIndex * 2 + 0 ] = boxMin;
        nodesAabbs[ nodeIndex * 2 + 1 ] = boxMax;

        sortedSpheres[ boxIndex ] = sphere;
    }

    __device__ int delta( int i, int j, uint32_t n, const uint64_t * const mortonCodes )
    {
        if ( j < 0 || j >= n )
            return -1;

        const uint64_t a = mortonCodes[ i ];
        const uint64_t b = mortonCodes[ j ];
        if ( a != b )
            return __clzll( a ^ b );
        else
            return __clzll( i ^ j ) + sizeof( uint64_t ) * 8;
    }

    template<typename T>
    __device__ int sgn( T val )
    {
        return ( T( 0 ) < val ) - ( val < T( 0 ) );
    }

    __global__ void construct( const uint32_t   count,
                               int * const      nodeParentIndices,
                               int * const      nodeLeftIndices,
                               int * const      nodeRightIndices,
                               uint64_t * const mortonCodes )
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i >= ( count - 1 ) )
            return;

        // Determine direction of the range (+1 or -1).
        const int d = sgn( delta( i, i + 1, count, mortonCodes ) - delta( i, i - 1, count, mortonCodes ) );

        // Compute upper bound for the length of the range.
        const int deltaMin = delta( i, i - d, count, mortonCodes );
        int       lmax     = 2;
        while ( delta( i, i + lmax * d, count, mortonCodes ) > deltaMin )
            lmax <<= 1;

        // Find the other end using binary search.
        int l = 0;
        for ( int t = lmax >> 1; t >= 1; t >>= 1 )
            if ( delta( i, i + ( l + t ) * d, count, mortonCodes ) > deltaMin )
                l += t;
        const int j = i + l * d;

        // Find the split position using binary search.
        const int deltaNode = delta( i, j, count, mortonCodes );
        int       s         = 0;
        int       k         = 2;
        int       t;
        do
        {
#define divCeil( a, b ) ( ( ( a ) + (b)-1 ) / ( b ) )
            t = divCeil( l, k );
#undef divCeil
            k <<= 1;
            if ( delta( i, i + ( s + t ) * d, count, mortonCodes ) > deltaNode )
                s += t;
        } while ( t > 1 );
        const int gamma = i + s * d + ::min( d, 0 );

        // Output child pointers.
        int left  = gamma;
        int right = gamma + 1;
        if ( min( i, j ) == gamma )
            left += count - 1;
        if ( max( i, j ) == gamma + 1 )
            right += count - 1;

        // Write node etc.
        nodeLeftIndices[ i ]  = left;
        nodeRightIndices[ i ] = right;

        // Parent indices.
        nodeParentIndices[ left ]  = i;
        nodeParentIndices[ right ] = i;

        if ( i == 0 )
            nodeParentIndices[ 0 ] = -1;
    }

    __global__ void refit( const uint32_t    nodeCount,
                           int *             terminationCounters,
                           const int * const nodeParentIndices,
                           const int * const nodeLeftIndices,
                           const int * const nodeRightIndices,
                           float4 * const    nodesAabbs,
                           uint32_t * const  nodesSizes )
    {
        const uint32_t leafIndex = blockDim.x * blockIdx.x + threadIdx.x + ( nodeCount >> 1 );
        if ( leafIndex >= nodeCount )
            return;

        nodesSizes[ leafIndex ] = 1;

        int nodeIndex = nodeParentIndices[ leafIndex ];
        while ( nodeIndex >= 0 && atomicAdd( &terminationCounters[ nodeIndex ], 1 ) > 0 )
        {
            // Sync. global memory writes.
            __threadfence();

            const int nodeLeftIndex  = nodeLeftIndices[ nodeIndex ];
            const int nodeRightIndex = nodeRightIndices[ nodeIndex ];

            const float3 leftMin  = make_float3( nodesAabbs[ nodeLeftIndex * 2 + 0 ] );
            const float3 leftMax  = make_float3( nodesAabbs[ nodeLeftIndex * 2 + 1 ] );
            const float3 rightMin = make_float3( nodesAabbs[ nodeRightIndex * 2 + 0 ] );
            const float3 rightMax = make_float3( nodesAabbs[ nodeRightIndex * 2 + 1 ] );

            apo::gpu::Aabbf aabb = apo::gpu::Aabbf::Degenerated();

            aabb.expand( leftMin );
            aabb.expand( leftMax );
            aabb.expand( rightMin );
            aabb.expand( rightMax );

            nodesAabbs[ nodeIndex * 2 + 0 ] = make_float4( aabb.min, 0.0f );
            nodesAabbs[ nodeIndex * 2 + 1 ] = make_float4( aabb.max, 0.0f );

            const uint32_t leftSize  = nodesSizes[ nodeLeftIndex ];
            const uint32_t rightSize = nodesSizes[ nodeRightIndex ];
            nodesSizes[ nodeIndex ]  = leftSize + rightSize;

            nodeIndex = nodeParentIndices[ nodeIndex ];

            // Ensure atomics execution order.
            __threadfence();
        }
    }

    void LBVH::build( uint32_t count, const Aabbf sceneBox, Real4 * spheres, bool sort )
    {
        elementCount      = count;
        indices           = DeviceBuffer::Typed<uint32_t>( count );
        nodeParentIndices = DeviceBuffer::Typed<int>( 2 * count - 1 );
        nodeLeftIndices   = DeviceBuffer::Typed<int>( 2 * count - 1 );
        nodeRightIndices  = DeviceBuffer::Typed<int>( 2 * count - 1 );
        nodesAabbs        = DeviceBuffer::Typed<float4>( 2 * ( 2 * count - 1 ) );
        nodesSizes        = DeviceBuffer::Typed<uint32_t>( 2 * count - 1, true );

        DeviceBuffer terminationCounters = DeviceBuffer::Typed<int>( count - 1, true );
        DeviceBuffer mortonCodes         = DeviceBuffer::Typed<uint64_t>( count );

        const float3 sceneMin = sceneBox.min;
        const float3 scale    = 1.f / ( sceneBox.max - sceneBox.min );

        auto [ gridDim, blockDim ] = KernelConfig::From( count, 256 );
        getMortonCodes<<<gridDim, blockDim>>>(
            count, sceneMin, scale, spheres, indices.get<uint32_t>(), mortonCodes.get<uint64_t>() );

        thrust::sort_by_key(
            thrust::device, mortonCodes.get<uint64_t>(), mortonCodes.get<uint64_t>() + count, indices.get<uint32_t>() );

        DeviceBuffer sortedSpheres = DeviceBuffer::Typed<Real4>( count );
        setLeaves<<<gridDim, blockDim>>>( count,
                                          spheres,
                                          indices.get<uint32_t>(),
                                          nodeLeftIndices.get<int>(),
                                          nodeRightIndices.get<int>(),
                                          nodesAabbs.get<float4>(),
                                          sortedSpheres.get<Real4>() );

        construct<<<gridDim, blockDim>>>( count,
                                          nodeParentIndices.get<int>(),
                                          nodeLeftIndices.get<int>(),
                                          nodeRightIndices.get<int>(),
                                          mortonCodes.get<uint64_t>() );

        auto [ nGridDim, nBlockDim ] = KernelConfig::From( 2 * count - 1, 256 );
        refit<<<nGridDim, nBlockDim>>>( 2 * count - 1,
                                        terminationCounters.get<int>(),
                                        nodeParentIndices.get<int>(),
                                        nodeLeftIndices.get<int>(),
                                        nodeRightIndices.get<int>(),
                                        nodesAabbs.get<float4>(),
                                        nodesSizes.get<uint32_t>() );

        if ( sort )
            copy( spheres, sortedSpheres.get<Real4>(), count );
    }

    void LBVH::buildf( uint32_t count, const Aabbf sceneBox, float4 * spheres, bool sort )
    {
        elementCount      = count;
        indices           = DeviceBuffer::Typed<uint32_t>( count );
        nodeParentIndices = DeviceBuffer::Typed<int>( 2 * count - 1 );
        nodeLeftIndices   = DeviceBuffer::Typed<int>( 2 * count - 1 );
        nodeRightIndices  = DeviceBuffer::Typed<int>( 2 * count - 1 );
        nodesAabbs        = DeviceBuffer::Typed<float4>( 2 * ( 2 * count - 1 ) );
        nodesSizes        = DeviceBuffer::Typed<uint32_t>( 2 * count - 1, true );

        DeviceBuffer terminationCounters = DeviceBuffer::Typed<int>( count - 1, true );
        DeviceBuffer mortonCodes         = DeviceBuffer::Typed<uint64_t>( count );

        const float3 sceneMin = sceneBox.min;
        const float3 scale    = 1.f / ( sceneBox.max - sceneBox.min );

        auto [ gridDim, blockDim ] = KernelConfig::From( count, 256 );
        getMortonCodesf<<<gridDim, blockDim>>>(
            count, sceneMin, scale, spheres, indices.get<uint32_t>(), mortonCodes.get<uint64_t>() );

        thrust::sort_by_key(
            thrust::device, mortonCodes.get<uint64_t>(), mortonCodes.get<uint64_t>() + count, indices.get<uint32_t>() );

        DeviceBuffer sortedSpheres = DeviceBuffer::Typed<float4>( count );
        setLeavesf<<<gridDim, blockDim>>>( count,
                                           spheres,
                                           indices.get<uint32_t>(),
                                           nodeLeftIndices.get<int>(),
                                           nodeRightIndices.get<int>(),
                                           nodesAabbs.get<float4>(),
                                           sortedSpheres.get<float4>() );

        construct<<<gridDim, blockDim>>>( count,
                                          nodeParentIndices.get<int>(),
                                          nodeLeftIndices.get<int>(),
                                          nodeRightIndices.get<int>(),
                                          mortonCodes.get<uint64_t>() );

        auto [ nGridDim, nBlockDim ] = KernelConfig::From( 2 * count - 1, 256 );
        refit<<<nGridDim, nBlockDim>>>( 2 * count - 1,
                                        terminationCounters.get<int>(),
                                        nodeParentIndices.get<int>(),
                                        nodeLeftIndices.get<int>(),
                                        nodeRightIndices.get<int>(),
                                        nodesAabbs.get<float4>(),
                                        nodesSizes.get<uint32_t>() );

        if ( sort )
            copy( spheres, sortedSpheres.get<float4>(), count );
    }
} // namespace apo::gpu