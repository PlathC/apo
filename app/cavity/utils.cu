#include <apo/core/utils.hpp>
#include <apo/gpu/algorithm.cuh>
#include <apo/gpu/lbvh.cuh>
#include <apo/gpu/topology_update.cuh>
#include <thrust/adjacent_difference.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

#include "optix/sphere_module.cuh"
#include "shaders/data.cuh"
#include "utils.hpp"

namespace apo::gpu
{
    apo::gpu::DeviceBuffer getHitRatios( const uint32_t           vertexNb,
                                         apo::gpu::DeviceBuffer & dVertices,
                                         ConstSpan<float>         sites )
    {
        // Loads the saved scene
        apo::optix::Context optixContext {};

        // Initialize OptiX pipelines
        apo::optix::GeometryPipeline pipeline { optixContext };
        pipeline.setPrimitiveType( OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE );

        apo::optix::Module rayGen { optixContext, "ptx/find_occlusion.ptx" };
        pipeline.setRayGen( rayGen, "__raygen__rg" );
        pipeline.setMiss( rayGen, "__miss__general" );

        apo::optix::SphereModule & sphereModule
            = pipeline.add( pipeline, "ptx/find_occlusion.ptx", "__closesthit__site" );
        apo::optix::SphereGeometry geometry { optixContext, sites };
        sphereModule.add( geometry );

        pipeline.compile();
        pipeline.updateGeometry();

        constexpr uint32_t OcclusionSampleNb = 64;
        auto               directions        = apo::gpu::getDirections( OcclusionSampleNb );

        apo::gpu::DeviceBuffer hitRatios = apo::gpu::DeviceBuffer::Typed<float>( vertexNb, true );

        // Trace
        apo::gpu::DeviceBuffer      dParameters = apo::gpu::DeviceBuffer::Typed<apo::OcclusionDetectionData>( 1 );
        apo::OcclusionDetectionData parameters {};
        parameters.handle     = pipeline.getHandle();
        parameters.vertexNb   = vertexNb;
        parameters.vertices   = dVertices.get<float4>();
        parameters.sampleNb   = OcclusionSampleNb;
        parameters.directions = directions.get<float3>();
        parameters.hitRatios  = hitRatios.get<float>();

        const auto sbt = pipeline.getBindingTable();
        apo::gpu::cudaCheck( cudaMemcpyAsync( dParameters.get(),
                                              &parameters,
                                              sizeof( apo::OcclusionDetectionData ),
                                              cudaMemcpyHostToDevice,
                                              optixContext.getStream() ) );

        pipeline.launch( dParameters.get(), sizeof( apo::OcclusionDetectionData ), sbt, vertexNb, 1, 1 );

        return hitRatios;
    }

    std::vector<float> getVertices( const uint32_t           vertexNb,
                                    apo::gpu::DeviceBuffer & dVertices,
                                    apo::gpu::DeviceBuffer & hitRatios,
                                    float                    hitRatioThreshold,
                                    float                    radiusThreshold )
    {
        apo::gpu::DeviceBuffer        dFinalVertices;
        apo::gpu::FilterConfiguration configuration { vertexNb, hitRatioThreshold, radiusThreshold };
        const uint32_t finalVertexNb = apo::gpu::filterVertices( configuration, dVertices, hitRatios, dFinalVertices );

        std::vector<float> vertices = {};
        vertices.resize( finalVertexNb * 4 );
        apo::gpu::cudaCheck( cudaMemcpy(
            vertices.data(), dFinalVertices.get(), sizeof( float ) * 4 * finalVertexNb, cudaMemcpyDeviceToHost ) );

        return vertices;
    }

    __global__ void getWriteableVertices( apo::gpu::ContextCellOriented context, uint32_t * const startWriteIndices )
    {
        const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= context.siteNb )
            return;

        const uint32_t vertexNb = context.vertexNb[ id ];
        const uint8_t  status   = context.status[ id ];
        if ( status == apo::gpu::CellStatus::Buried )
        {
            startWriteIndices[ id ] = 0;
            return;
        }

        uint32_t writeableVertexNb = 0;
        for ( uint32_t v = 0; v < vertexNb; v++ )
        {
            const apo::gpu::Vertex vertex = context.vertices[ id * apo::gpu::impl::TotalMaxVertexNb + v ];
            if ( id > vertex.j || vertex.j >= context.siteNb || vertex.k >= context.siteNb
                 || vertex.l >= context.siteNb )
                continue;

            writeableVertexNb++;
        }

        startWriteIndices[ id ] = writeableVertexNb;
    }

    __global__ void saveVertices( apo::gpu::ContextCellOriented context,
                                  const uint32_t * const __restrict__ startWriteIndices,
                                  float4 * __restrict__ output )
    {
        const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= context.siteNb )
            return;

        const uint8_t status = context.status[ id ];
        if ( status == apo::gpu::CellStatus::Buried )
            return;

        const uint32_t start    = startWriteIndices[ id ];
        const uint32_t vertexNb = context.vertexNb[ id ];

        uint32_t currentWritingIndex = 0;
        for ( uint32_t v = 0; v < vertexNb; v++ )
        {
            const apo::gpu::Vertex vertex = context.vertices[ id * apo::gpu::impl::TotalMaxVertexNb + v ];
            if ( id > vertex.j || vertex.j >= context.siteNb || vertex.k >= context.siteNb
                 || vertex.l >= context.siteNb )
                continue;

            const apo::gpu::Real4 si = context.sites[ id ];
            const apo::gpu::Real4 sj = context.sites[ vertex.j ];
            const apo::gpu::Real4 sk = context.sites[ vertex.k ];
            const apo::gpu::Real4 sl = context.sites[ vertex.l ];

            apo::gpu::Real4 x[ 2 ];
            apo::gpu::quadrisector( si, sj, sk, sl, x[ 0 ], x[ 1 ] );

            if ( !( id < vertex.j && id < vertex.k && id < vertex.l && vertex.j < vertex.k && vertex.j < vertex.l
                    && vertex.k < vertex.l ) )
                printf( "Error: unsorted edge {%d, %d, %d, %d}\n", id, vertex.j, vertex.k, vertex.l );

            output[ start + currentWritingIndex ] = apo::gpu::toFloat( x[ vertex.type ] );
            currentWritingIndex++;
        }
    }

    uint32_t getVertices( apo::ConstSpan<apo::Real> sites, apo::gpu::DeviceBuffer & dVertices )
    {
        apo::gpu::AlgorithmGPU algorithm { sites };
        algorithm.build();

        apo::gpu::ContextCellOriented context {};
        context.sites     = algorithm.dSites.get<apo::gpu::Real4>();
        context.siteNb    = algorithm.m_siteNb;
        context.minRadius = algorithm.m_minRadius;
        context.maxRadius = algorithm.m_maxRadius;

        // Storage for data structure
        context.status   = algorithm.dStatus.get<uint8_t>();
        context.vertexNb = algorithm.dVerticesNb.get<uint32_t>();
        context.vertices = algorithm.dVertices.get<apo::gpu::Vertex>();
        context.edges    = algorithm.dEdges.get<apo::gpu::Edge>();
        context.edgeNb   = algorithm.dEdgesNb.get<uint32_t>();

        apo::gpu::DeviceBuffer writeIndices = apo::gpu::DeviceBuffer::Typed<uint32_t>( algorithm.m_siteNb );
        auto [ gridDim, blockDim ]          = apo::gpu::KernelConfig::From( context.siteNb, 256 );
        getWriteableVertices<<<gridDim, blockDim>>>( context, writeIndices.get<uint32_t>() );
        apo::gpu::cudaCheck( "getWriteableVertices" );

        uint32_t lastVertexNb = 0;
        apo::gpu::mmemcpy<apo::gpu::MemcpyType::DeviceToHost>(
            &lastVertexNb, writeIndices.get<uint32_t>() + algorithm.m_siteNb - 1, 1 );

        thrust::exclusive_scan( thrust::device,
                                writeIndices.get<uint32_t>(),
                                writeIndices.get<uint32_t>() + algorithm.m_siteNb,
                                writeIndices.get<uint32_t>() );

        uint32_t lastWritingIndex = 0;
        apo::gpu::mmemcpy<apo::gpu::MemcpyType::DeviceToHost>(
            &lastWritingIndex, writeIndices.get<uint32_t>() + algorithm.m_siteNb - 1, 1 );

        const uint32_t vertexNb = lastWritingIndex + lastVertexNb;
        if ( vertexNb == 0 )
            return 0;

        dVertices = apo::gpu::DeviceBuffer::Typed<float4>( vertexNb );
        saveVertices<<<gridDim, blockDim>>>( context, writeIndices.get<uint32_t>(), dVertices.get<float4>() );

        return vertexNb;
    }

    // Tien-Tsin Wong et al., Sampling with Hammersley and Halton Points
    // Reference: https://www.cse.cuhk.edu.hk/~ttwong/papers/udpoint/udpoint.pdf
    inline std::vector<float3> sphereHalton( int n, int p2 )
    {
        auto result = std::vector<float3>( n );
        for ( int k = 0; k < n; k++ )
        {
            float t  = 0.f;
            int   kk = k;
            for ( float p = 0.5f; kk; p *= .5f, kk >>= 1 )
            {
                if ( kk & 1 )
                {
                    t += p;
                }
            }

            t = 2.f * t - 1.f;

            float st  = std::sqrt( 1.f - t * t );
            float phi = 0.f;
            float ip  = 1.f / p2;
            kk        = k;
            for ( float p = ip; kk; p *= ip, kk /= p2 )
            {
                int a = kk % p2;
                if ( a )
                {
                    phi += a * p;
                }
            }

            float phirad = phi * 4.f * apo::Pi;
            result[ k ]  = { st * std::cos( phirad ), st * std::sin( phirad ), t };
        }

        return result;
    }

    apo::gpu::DeviceBuffer getDirections( uint32_t count )
    {
        const std::vector<float3> direction = sphereHalton( static_cast<int>( count + 1 ), 3 );
        return apo::gpu::DeviceBuffer::From<float3>( direction );
    }

    __global__ void filterByHitRatios( FilterConfiguration configuration,
                                       const float * const __restrict__ hitRatios,
                                       uint8_t * __restrict__ status )
    {
        const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i >= configuration.vertexNb )
            return;

        status[ i ] &= hitRatios[ i ] > configuration.hitRatioThreshold;
    }

    __global__ void filterByRadius( FilterConfiguration configuration,
                                    const float4 * __restrict__ const vertices,
                                    uint8_t * __restrict__ status )
    {
        const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i >= configuration.vertexNb )
            return;

        status[ i ] &= vertices[ i ].w > configuration.radiusThreshold;
    }

    __global__ void fillEdges( uint32_t vertexNb,
                               const uint32_t * const __restrict__ verticesIds,
                               uint4 * __restrict__ edges )
    {
        const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= vertexNb )
            return;

        const uint32_t i = verticesIds[ id * 4 + 0 ];
        const uint32_t j = verticesIds[ id * 4 + 1 ];
        const uint32_t k = verticesIds[ id * 4 + 2 ];
        const uint32_t l = verticesIds[ id * 4 + 3 ];

        if ( !( i < j && i < k && i < l && j < k && j < l ) )
            printf( "Error: unsorted edge {%d, %d, %d, %d}\n", i, j, k, l );

        edges[ id * 4 + 0 ] = make_uint4( i, j, k, id );
        edges[ id * 4 + 1 ] = make_uint4( i, k, l, id );
        edges[ id * 4 + 2 ] = make_uint4( i, j, l, id );
        edges[ id * 4 + 3 ] = make_uint4( j, k, l, id );
    }

    __global__ void findStartEdges( const uint32_t edgeNb,
                                    const uint4 * const __restrict__ edges,
                                    uint8_t * __restrict__ isFirst )
    {
        const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= edgeNb )
            return;

        if ( id == 0 )
        {
            isFirst[ id ] = true;
            return;
        }

        const uint4 previous = edges[ id - 1 ];
        const uint4 current  = edges[ id ];
        isFirst[ id ]        = previous.x != current.x || previous.y != current.y || previous.z != current.z;
    }

    __global__ void saveVerticesIds( const uint32_t edgeNb, const uint4 * edges, uint32_t * __restrict__ verticesIds )
    {
        const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= edgeNb )
            return;

        verticesIds[ id ] = edges[ id ].w;
    }

    __global__ void saveVerticesNb( const uint32_t edgeNb,
                                    const uint32_t * const __restrict__ vertexNb,
                                    uint4 * __restrict__ edges )
    {
        const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= edgeNb )
            return;

        edges[ id ].w = vertexNb[ id ];
    }

    __global__ void filterEdgesByMinimas( const uint32_t      edgeNb,
                                          FilterConfiguration configuration,
                                          const uint4 * const __restrict edges,
                                          const Real4 * const __restrict__ sites,
                                          uint32_t * __restrict__ status )
    {
        const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= edgeNb )
            return;

        const uint4           edge = edges[ id ];
        const apo::gpu::Real4 si   = sites[ edge.x ];
        const apo::gpu::Real4 sj   = sites[ edge.y ];
        const apo::gpu::Real4 sk   = sites[ edge.z ];

        apo::gpu::Real4 x[ 2 ];
        const uint8_t   count = apo::gpu::trisectorVertex( si, sj, sk, x[ 0 ], x[ 1 ] );

        status[ id ] = count == 1 && x[ 0 ].w > configuration.radiusThreshold;
    }

    __global__ void filterByNeighbors( FilterConfiguration configuration,
                                       LBVH::View          bvh,
                                       const float4 * __restrict__ const vertices,
                                       uint8_t * __restrict__ status )
    {
        const uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
        if ( v >= configuration.vertexNb || !status[ v ] )
            return;

        const float4 vertex = vertices[ v ];

        int2     stack[ 32 ];
        int2 *   stackPtr = stack;
        uint32_t nodeId   = 0; // Starting from the root

        uint32_t invalidating = 0xffffffff;
        while ( true )
        {
            while ( true )
            {
                const bool isLeaf = nodeId >= bvh.count - 1;
                if ( isLeaf )
                    break;

                const uint32_t left  = bvh.nodesLeft[ nodeId ];
                const uint32_t right = bvh.nodesRight[ nodeId ];

                const Aabbf leftBB = {
                    make_float3( bvh.nodesAabb[ left * 2 + 0 ] ),
                    make_float3( bvh.nodesAabb[ left * 2 + 1 ] ),
                };
                const Aabbf rightBB = {
                    make_float3( bvh.nodesAabb[ right * 2 + 0 ] ),
                    make_float3( bvh.nodesAabb[ right * 2 + 1 ] ),
                };

                const float dLeft  = leftBB.distance( vertex );
                const float dRight = rightBB.distance( vertex );

                const bool     isLeftClosest = dLeft < dRight;
                const float    minFarthest   = isLeftClosest ? dRight : dLeft;
                const uint32_t farthestChild = isLeftClosest ? right : left;
                const float    minClosest    = isLeftClosest ? dLeft : dRight;
                const uint32_t closestChild  = isLeftClosest ? left : right;

                const bool exploreFarthest = minFarthest < 0.f;
                const bool exploreClosest  = minClosest < 0.f;
                if ( !exploreFarthest && !exploreClosest )
                    break;

                nodeId = exploreClosest ? closestChild : farthestChild;
                if ( exploreClosest && exploreFarthest )
                {
                    const int distBitsFar = __float_as_int( minFarthest );

                    *stackPtr = make_int2( farthestChild, distBitsFar );
                    stackPtr++;
                }
            }

            const bool isLeaf = nodeId >= bvh.count - 1 && nodeId < 2 * bvh.count - 1;
            if ( isLeaf )
            {
                const uint32_t otherId = bvh.indices[ nodeId - ( bvh.count - 1 ) ];
                if ( otherId != v && vertices[ otherId ].w > vertex.w )
                {
                    const float4 vertex2  = vertices[ otherId ];
                    const float  distance = sphereDistance( vertex, vertex2 );

                    const float  d = dot( make_float3( vertex ) - make_float3( vertex2 ),
                                         make_float3( vertex ) - make_float3( vertex2 ) );
                    const float  t = ( vertex.w * vertex.w - vertex2.w * vertex2.w + d ) / ( 2.f * d );
                    const float3 c = make_float3( vertex ) + ( make_float3( vertex2 ) - make_float3( vertex ) ) * t;

                    const float sqCircleDistance = dot( make_float3( vertex ) - c, make_float3( vertex ) - c );
                    const float r                = ::sqrtf( vertex.w * vertex.w - sqCircleDistance );

                    if ( apo::lessThan( r, configuration.radiusThreshold ) )
                        invalidating = otherId;
                }
            }

            if ( invalidating != 0xffffffff )
                break;

            bool continueTraversal = true;
            while ( true )
            {
                if ( stackPtr == stack )
                {
                    continueTraversal = false;
                    break;
                }

                --stackPtr;
                if ( __int_as_float( stackPtr->y ) > 0.f )
                    continue;

                nodeId = stackPtr->x;
                break;
            }

            if ( !continueTraversal )
                break;
        }

        status[ v ] &= invalidating == 0xffffffff;
    }

    __global__ void filterEdgesByVertexNb( const uint32_t edgeNb,
                                           const uint4 * const __restrict edges,
                                           uint32_t * __restrict__ status )
    {
        const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= edgeNb )
            return;

        status[ id ] = edges[ id ].w == 2;
    }

    __device__ __host__ inline bool operator==( uint3 a, uint3 b ) { return a.x == b.x && a.y == b.y && a.z == b.z; }

    __device__ __host__ inline bool operator!=( uint3 a, uint3 b ) { return a.x != b.x || a.y != b.y || a.z != b.z; }

    __global__ void writeFinalEdges( const uint32_t edgeNb,
                                     const uint32_t * const __restrict__ writeIds,
                                     const uint4 * const __restrict__ baseEdges,
                                     const uint4 * const __restrict__ verticesIds,
                                     const uint32_t * const __restrict__ baseEdgesStartVerticesId,
                                     const uint32_t * const __restrict__ edgeVerticesIds,
                                     const uint32_t * const __restrict__ status,
                                     uint4 * __restrict__ finalEdges,
                                     uint2 * __restrict__ finalEdgesVertexIds )
    {
        const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= edgeNb )
            return;

        if ( !status[ id ] )
            return;

        uint4 edge = baseEdges[ id ];
        edge.w     = 0;

        finalEdges[ writeIds[ id ] ] = edge;

        const uint32_t start = baseEdgesStartVerticesId[ id ];
        finalEdgesVertexIds[ writeIds[ id ] ]
            = make_uint2( edgeVerticesIds[ start + 0 ], edgeVerticesIds[ start + 1 ] );

        const uint4 a = verticesIds[ edgeVerticesIds[ start + 0 ] ];
        const uint4 b = verticesIds[ edgeVerticesIds[ start + 1 ] ];

        const uint3 a1 = make_uint3( a.x, a.y, a.z );
        const uint3 a2 = make_uint3( a.x, a.z, a.w );
        const uint3 a3 = make_uint3( a.x, a.y, a.w );
        const uint3 a4 = make_uint3( a.y, a.z, a.w );

        const uint3 b1 = make_uint3( b.x, b.y, b.z );
        const uint3 b2 = make_uint3( b.x, b.z, b.w );
        const uint3 b3 = make_uint3( b.x, b.y, b.w );
        const uint3 b4 = make_uint3( b.y, b.z, b.w );

        if ( ( a1 != b1 && a1 != b2 && a1 != b3 && a1 != b4 ) && ( a2 != b1 && a2 != b2 && a2 != b3 && a2 != b4 )
             && ( a3 != b1 && a3 != b2 && a3 != b3 && a3 != b4 ) && ( a4 != b1 && a4 != b2 && a4 != b3 && a4 != b4 ) )
            printf( "%d - Error: v(%d) {%d, %d, %d, %d} is not on same edge as v(%d) {%d, %d, %d, %d}\n",
                    id,
                    edgeVerticesIds[ start + 0 ],
                    a.x,
                    a.y,
                    a.z,
                    a.w,
                    edgeVerticesIds[ start + 1 ],
                    b.x,
                    b.y,
                    b.z,
                    b.w );
    }

    uint32_t filterVertices( FilterConfiguration      configuration,
                             apo::gpu::DeviceBuffer & vertices,
                             apo::gpu::DeviceBuffer & hitRatios,
                             apo::gpu::DeviceBuffer & finalVertices )
    {
        // Mask
        apo::gpu::DeviceBuffer status = apo::gpu::DeviceBuffer::Typed<uint8_t>( configuration.vertexNb, false );
        apo::gpu::cudaCheck( cudaMemset( status.get<uint8_t>(), 0xff, configuration.vertexNb * sizeof( uint8_t ) ) );

        // Vertex filtering
        auto [ gridDim, blockDim ] = apo::gpu::KernelConfig::From( configuration.vertexNb, 256 );
        filterByHitRatios<<<gridDim, blockDim>>>( configuration, hitRatios.get<float>(), status.get<uint8_t>() );
        filterByRadius<<<gridDim, blockDim>>>( configuration, vertices.get<float4>(), status.get<uint8_t>() );

        // Compaction
        const uint32_t finalVertexNb = thrust::count_if( thrust::device,
                                                         status.get<uint8_t>(),
                                                         status.get<uint8_t>() + configuration.vertexNb,
                                                         thrust::identity<uint8_t>() );

        finalVertices = apo::gpu::DeviceBuffer::Typed<float4>( finalVertexNb );
        using namespace thrust::placeholders;
        thrust::copy_if( //
            thrust::device,
            vertices.get<float4>(),
            vertices.get<float4>() + configuration.vertexNb,
            status.get<uint8_t>(),
            finalVertices.get<float4>(),
            thrust::identity<uint8_t>() );

        return finalVertexNb;
    }
} // namespace apo::gpu
