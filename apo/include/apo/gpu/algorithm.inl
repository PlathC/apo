#include <set>

#include <cub/warp/warp_merge_sort.cuh>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include "apo/core/logger.hpp"
#include "apo/core/utils.hpp"
#include "apo/gpu/algorithm.cuh"
#include "apo/gpu/benchmark.cuh"
#include "apo/gpu/lbvh.cuh"
#include "apo/gpu/math.cuh"
#include "apo/gpu/memory.cuh"
#include "apo/gpu/topology_update.cuh"
#include "apo/gpu/utils.cuh"

namespace apo::gpu
{
    namespace impl
    {
        constexpr static uint32_t TotalMaxVertexNb = 152;
        constexpr static uint32_t TotalMaxEdgeNb   = 264;
        constexpr static uint32_t KnnBlockSize     = 256;
        constexpr static uint32_t BlockSize        = 128;
        constexpr static uint32_t MaxNewEdgeNb     = 64;
        constexpr static uint32_t MaxBisectorNb    = 512;

        template<uint32_t K>
        __global__ void getKnns( const uint32_t            iteration,
                                 const ContextCellOriented context,
                                 const LBVH::View          bvh,
                                 uint32_t * __restrict__ knns )
        {
            __shared__ uint32_t sharedKnns[ KnnBlockSize * K ];
            __shared__ float    sharedKnnsDistance[ KnnBlockSize * K ];

            const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if ( id >= context.siteNb )
                return;

            const float4 si = apo::gpu::toFloat( context.sites[ id ] );

            float          startingDistance = -MaxFloat;
            const uint32_t last             = knns[ id * K + K - 1 ];
            if ( iteration != 0 && last < context.siteNb )
            {
                const float4 previousData = toFloat( context.sites[ last ] );
                startingDistance          = apo::gpu::sphereDistance( previousData, si );
            }
            else if ( iteration != 0 ) // No more neighbor to fetch
            {
                for ( uint32_t kk = 0; kk < K; kk++ )
                    knns[ id * K + kk ] = 0xffffffff;

                return;
            }

            uint32_t * localKnns         = sharedKnns + threadIdx.x * K;
            float *    localKnnsDistance = sharedKnnsDistance + threadIdx.x * K;
            for ( uint32_t kk = 0; kk < K; kk++ )
            {
                localKnns[ kk ]         = 0xffffffff;
                localKnnsDistance[ kk ] = MaxFloat;
            }

            bvh.getKnns( K,
                         context.sites,
                         localKnns,
                         localKnnsDistance,
                         si,
                         startingDistance,
                         [ id ]( const uint32_t j ) { return id != j; } );

            // Write result to global memory
            for ( uint8_t kk = 0; kk < K; kk++ )
                knns[ id * K + kk ] = localKnns[ kk ];

            if ( iteration > 0 )
                return;

            const Real4 closest = context.sites[ localKnns[ 0 ] ];
            const Real  ri      = si.w;
            const Real  rj      = closest.w;
            const Real  bigger  = apo::max( ri, rj );
            const Real  smaller = apo::min( ri, rj );

            if ( apo::lessThan( length( Real3 { si.x, si.y, si.z } - Real3 { closest.x, closest.y, closest.z } ),
                                bigger - smaller ) )
            {
                if ( apo::lessThan( rj, ri ) )
                    return;

                context.status[ id ] = CellStatus::Buried;
            }
        }

        inline __global__ void getCells( ContextCellOriented    context,
                                         const uint32_t         k,
                                         const uint32_t * const knns,
                                         float * __restrict__ knownRadii )
        {
            union TempStorage
            {
                WarpMergeSortNewEdgeT<MaxNewEdgeNb>::TempStorage newEdgeSort;
            };

            constexpr uint16_t     WarpPerBlock = BlockSize / WarpSize;
            __shared__ TempStorage tempStorages[ WarpPerBlock ];
            __shared__ Edge        edges[ TotalMaxEdgeNb * WarpPerBlock ];
            __shared__ Vertex      vertices[ TotalMaxVertexNb * WarpPerBlock ];

            const auto     block  = cg::this_thread_block();
            const auto     warp   = cg::tiled_partition<WarpSize>( block );
            const uint32_t warpId = threadIdx.x / WarpSize;

            const uint32_t start = blockIdx.x * WarpPerBlock;
            const uint32_t end   = apo::min( ( blockIdx.x + 1 ) * WarpPerBlock, context.siteNb );

            TempStorage & tempStorage = tempStorages[ warp.meta_group_rank() ];
            for ( uint32_t i = start + warpId; i < end; i += WarpPerBlock )
            {
                const uint8_t status = context.status[ i ];
                if ( status == CellStatus::FullyValidated || status == CellStatus::Buried )
                    continue;

                // Load edges and find neighbor list based on edges
                const uint32_t totalEdgeNb = context.edgeNb[ i ];
                CellEdges      cellEdges   = {
                    totalEdgeNb,
                    edges + TotalMaxEdgeNb * warp.meta_group_rank(),
                };

                for ( uint32_t e = warp.thread_rank(); e < totalEdgeNb; e += warpSize )
                {
                    const Edge edge      = context.edges[ i * TotalMaxEdgeNb + e ];
                    cellEdges.edges[ e ] = edge;
                }

                // Load vertices
                const uint32_t totalVertexNb = context.vertexNb[ i ];
                CellVertices   cellVertices  = {
                    totalVertexNb,
                    vertices + TotalMaxVertexNb * warp.meta_group_rank(),
                };

                for ( uint32_t v = warp.thread_rank(); v < totalVertexNb; v += warpSize )
                {
                    const Vertex vertex        = context.vertices[ i * TotalMaxVertexNb + v ];
                    cellVertices.vertices[ v ] = vertex;
                }

                // Synchronized for shared memory write
                warp.sync();

                const Real4 si          = context.sites[ i ];
                Real        knownRadius = -apo::Max; // Save reached distance to validate cell topology

                for ( uint32_t nId = 0; nId < k; nId++ )
                {
                    const uint32_t j = knns[ i * k + nId ];
                    if ( j == 0xffffffff )
                        break;

                    if ( context.status[ j ] == CellStatus::Buried )
                        continue;

                    const Real4 sj          = context.sites[ j ];
                    const bool  contributed = updateCell<TotalMaxVertexNb, TotalMaxEdgeNb, MaxNewEdgeNb>( //
                        warp,
                        context,
                        i,
                        si,
                        cellVertices,
                        cellEdges,
                        j,
                        sj,
                        tempStorage.newEdgeSort );

                    knownRadius = apo::max( knownRadius, apo::gpu::sphereDistance( si, sj ) );
                }

                // Check if the cell is completely validated from the security radius
                const Real maxRadius      = cellRadius( warp, context, si, cellVertices, cellEdges );
                const Real securityRadius = apo::Real( 2 ) * maxRadius;
                if ( warp.thread_rank() == 0 && apo::greaterThan( knownRadius, securityRadius ) )
                    context.status[ i ] = CellStatus::FullyValidated;

                if ( threadIdx.x % warpSize == 0 )
                {
                    context.vertexNb[ i ] = cellVertices.size;
                    context.edgeNb[ i ]   = cellEdges.size;

                    knownRadii[ i ] = static_cast<float>( knownRadius );
                }

                for ( uint32_t e = warp.thread_rank(); e < cellEdges.size; e += warpSize )
                    context.edges[ i * TotalMaxEdgeNb + e ] = cellEdges.edges[ e ];

                for ( uint32_t v = warp.thread_rank(); v < cellVertices.size; v += warpSize )
                    context.vertices[ i * TotalMaxVertexNb + v ] = cellVertices.vertices[ v ];
            }
        }

        inline __global__ void globalVertexValidation( ContextCellOriented    context,
                                                       LBVH::View             bvh,
                                                       const uint32_t         taskNb,
                                                       const uint32_t * const tasks,
                                                       const float * const    knownRadii )
        {
            union TempStorage
            {
                WarpMergeSortNewEdgeT<MaxNewEdgeNb>::TempStorage newEdgeSort;
            };

            constexpr uint16_t     WarpPerBlock = BlockSize / WarpSize;
            __shared__ TempStorage tempStorages[ WarpPerBlock ];
            __shared__ Edge        edges[ TotalMaxEdgeNb * WarpPerBlock ];
            __shared__ Vertex      vertices[ TotalMaxVertexNb * WarpPerBlock ];

            const auto     block  = cg::this_thread_block();
            const auto     warp   = cg::tiled_partition<WarpSize>( block );
            const uint32_t warpId = threadIdx.x / WarpSize;

            const uint32_t start = blockIdx.x * WarpPerBlock;
            const uint32_t end   = apo::min( ( blockIdx.x + 1 ) * WarpPerBlock, taskNb );

            TempStorage & tempStorage = tempStorages[ warp.meta_group_rank() ];
            for ( uint32_t t = start + warpId; t < end; t += WarpPerBlock )
            {
                const uint32_t i      = tasks[ t ];
                const uint8_t  status = context.status[ i ];
                if ( status == CellStatus::FullyValidated || status == CellStatus::Buried )
                    continue;

                // Load edges
                const uint32_t totalEdgeNb = context.edgeNb[ i ];
                CellEdges      cellEdges   = {
                    totalEdgeNb,
                    edges + TotalMaxEdgeNb * warp.meta_group_rank(),
                };

                for ( uint32_t e = warp.thread_rank(); e < totalEdgeNb; e += warpSize )
                {
                    const Edge edge      = context.edges[ i * TotalMaxEdgeNb + e ];
                    cellEdges.edges[ e ] = edge;
                }

                // Load vertices
                const float    knownRadius   = knownRadii[ i ];
                const uint32_t totalVertexNb = context.vertexNb[ i ];
                CellVertices   cellVertices  = {
                    totalVertexNb,
                    vertices + TotalMaxVertexNb * warp.meta_group_rank(),
                };

                for ( uint32_t v = warp.thread_rank(); v < totalVertexNb; v += warpSize )
                {
                    const Vertex vertex        = context.vertices[ i * TotalMaxVertexNb + v ];
                    cellVertices.vertices[ v ] = vertex;
                }

                warp.sync(); // Shared memory write

                uint32_t    vs = 0;
                const Real4 si = context.sites[ i ];

                std::size_t restartNb = 0;
                while ( vs < ( cellVertices.size / WarpSize ) + 1 && restartNb < 100 )
                {
                    const uint32_t vid        = vs * WarpSize + warp.thread_rank();
                    bool           needSearch = vid < cellVertices.size;
                    vs++;

                    Vertex * vertex = nullptr;
                    if ( needSearch )
                    {
                        vertex     = &cellVertices.vertices[ vid ];
                        needSearch = vertex->status == 0;
                    }

                    float4 fCurrent;
                    Real4  current;
                    if ( needSearch )
                    {
                        const Real4 sj = context.sites[ vertex->j ];
                        const Real4 sk = context.sites[ vertex->k ];
                        const Real4 sl = context.sites[ vertex->l ];

                        Real4 x[ 2 ];
                        apo::gpu::quadrisector( si, sj, sk, sl, x[ 0 ], x[ 1 ] );
                        current  = x[ vertex->type ];
                        fCurrent = toFloat( current );

                        if ( apo::Real( 2 ) * current.w < knownRadius )
                        {
                            needSearch     = false;
                            vertex->status = 1;
                        }
                    }

                    int2     stack[ 32 ];
                    int2 *   stackPtr = stack;
                    uint32_t nodeId   = 0; // Starting from the root

                    float    neighborDistance  = apo::MaxFloat;
                    uint32_t neighbor          = 0xffffffff;
                    bool     continueTraversal = needSearch;
                    while ( true )
                    {
                        while ( continueTraversal )
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

                            const float dLeft  = leftBB.distance( fCurrent );
                            const float dRight = rightBB.distance( fCurrent );

                            const bool     isLeftClosest = dLeft < dRight;
                            const float    minFarthest   = isLeftClosest ? dRight : dLeft;
                            const uint32_t farthestChild = isLeftClosest ? right : left;
                            const float    minClosest    = isLeftClosest ? dLeft : dRight;
                            const uint32_t closestChild  = isLeftClosest ? left : right;

                            const bool exploreFarthest = minFarthest < neighborDistance;
                            const bool exploreClosest  = minClosest < neighborDistance;
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

                        const bool     isLeafValid   = nodeId >= bvh.count - 1 && nodeId < ( 2 * bvh.count - 1 );
                        const uint32_t j             = nodeId - ( bvh.count - 1 );
                        bool           canContribute = isLeafValid && continueTraversal && j != i && j != vertex->j
                                             && j != vertex->k && j != vertex->l
                                             && context.status[ j ] != CellStatus::Buried;

                        apo::gpu::forEach(
                            warp.ballot( canContribute ),
                            [ & ]( const uint8_t tid )
                            {
                                const uint32_t currentJ = warp.shfl( j, tid );

                                bool           isCurrentKnown = false;
                                const uint32_t edgeSteps      = ( cellEdges.size / WarpSize ) + 1;
                                for ( uint8_t es = 0; es < edgeSteps && !isCurrentKnown; es++ )
                                {
                                    const uint32_t eid  = warp.thread_rank() + WarpSize * es;
                                    const Edge &   edge = cellEdges.edges[ eid ];
                                    isCurrentKnown      = warp.any( //
                                        eid < cellEdges.size && ( edge.j == currentJ || edge.k == currentJ ) );
                                }

                                if ( warp.thread_rank() == tid )
                                    canContribute = !isCurrentKnown;
                            } );

                        if ( canContribute )
                        {
                            const Real4  sj = context.sites[ j ];
                            const float3 cj = make_float3( toFloat( sj ) );

                            const float distance = apo::gpu::sphereDistance( toFloat( sj ), fCurrent );
                            if ( distance < neighborDistance )
                            {
                                neighborDistance = distance;
                                neighbor         = j;
                            }
                        }

                        while ( continueTraversal )
                        {
                            if ( stackPtr == stack )
                            {
                                continueTraversal = false;
                                break;
                            }

                            --stackPtr;
                            if ( __int_as_float( stackPtr->y ) > neighborDistance )
                                continue;

                            nodeId = stackPtr->x;
                            break;
                        }

                        const uint32_t continueSearch = warp.ballot( continueTraversal );
                        if ( !continueSearch )
                            break;
                    }

                    const bool isValid = neighbor == 0xffffffff //
                                         || !intersect( current, context.sites[ neighbor ] );
                    if ( needSearch )
                        vertex->status = 1;

                    const uint32_t cellChanged = warp.ballot( !isValid );
                    apo::gpu::forEach(
                        cellChanged,
                        [ & ]( const uint8_t tid )
                        {
                            const uint32_t currentJ       = warp.shfl( neighbor, tid );
                            bool           isCurrentKnown = false;

                            // Test again since the data structure may have been modified
                            const uint32_t edgeSteps = ( cellEdges.size / WarpSize ) + 1;
                            for ( uint8_t es = 0; es < edgeSteps && !isCurrentKnown; es++ )
                            {
                                const uint32_t eid  = warp.thread_rank() + WarpSize * es;
                                const Edge &   edge = cellEdges.edges[ eid ];
                                isCurrentKnown      = warp.any( //
                                    eid < cellEdges.size && ( edge.j == currentJ || edge.k == currentJ ) );
                            }

                            if ( isCurrentKnown )
                                return;

                            const Real4 sj = context.sites[ currentJ ];
                            updateCell<TotalMaxVertexNb, TotalMaxEdgeNb, MaxNewEdgeNb>( //
                                warp,
                                context,
                                i,
                                si,
                                cellVertices,
                                cellEdges,
                                currentJ,
                                sj,
                                tempStorage.newEdgeSort );
                        } );

                    // If cell changed, we can't ensure that new vertices are not positioned at the start of the buffer
                    // Then, we restart validation.
                    if ( cellChanged )
                    {
                        vs = 0;
                        restartNb++;
                    }
                }

                if ( threadIdx.x % warpSize == 0 )
                {
                    context.vertexNb[ i ] = cellVertices.size;
                    context.edgeNb[ i ]   = cellEdges.size;
                }

                for ( uint32_t e = warp.thread_rank(); e < cellEdges.size; e += warpSize )
                    context.edges[ i * TotalMaxEdgeNb + e ] = cellEdges.edges[ e ];

                for ( uint32_t v = warp.thread_rank(); v < cellVertices.size; v += warpSize )
                    context.vertices[ i * TotalMaxVertexNb + v ] = cellVertices.vertices[ v ];
            }
        }

        inline __global__ void globalEdgeValidation( ContextCellOriented    context,
                                                     LBVH::View             bvh,
                                                     const uint32_t         taskNb,
                                                     const uint32_t * const tasks,
                                                     const float * const    knownRadii )
        {
            union TempStorage
            {
                WarpMergeSortNewEdgeT<MaxNewEdgeNb>::TempStorage newEdgeSort;
            };

            constexpr uint16_t     WarpPerBlock = BlockSize / WarpSize;
            __shared__ TempStorage tempStorages[ WarpPerBlock ];
            __shared__ Edge        edges[ TotalMaxEdgeNb * WarpPerBlock ];
            __shared__ Vertex      vertices[ TotalMaxVertexNb * WarpPerBlock ];

            const auto     block  = cg::this_thread_block();
            const auto     warp   = cg::tiled_partition<WarpSize>( block );
            const uint32_t warpId = threadIdx.x / WarpSize;

            const uint32_t start = blockIdx.x * WarpPerBlock;
            const uint32_t end   = apo::min( ( blockIdx.x + 1 ) * WarpPerBlock, taskNb );

            TempStorage & tempStorage = tempStorages[ warp.meta_group_rank() ];
            for ( uint32_t t = start + warpId; t < end; t += WarpPerBlock )
            {
                const uint32_t i      = tasks[ t ];
                const uint8_t  status = context.status[ i ];
                if ( status == CellStatus::FullyValidated || status == CellStatus::Buried )
                    continue;

                const float knownRadius = knownRadii[ i ];

                // Load edges
                const uint32_t totalEdgeNb = context.edgeNb[ i ];
                CellEdges      cellEdges   = {
                    totalEdgeNb,
                    edges + TotalMaxEdgeNb * warp.meta_group_rank(),
                };

                for ( uint32_t e = warp.thread_rank(); e < totalEdgeNb; e += warpSize )
                {
                    Edge edge = context.edges[ i * TotalMaxEdgeNb + e ];

                    // Allow to differentiate between edges that were there before edge validation and the
                    // potential new ones added during edge validation. At this stage, we only validate edge that
                    // have been vertex validated
                    if ( edge.status != 1 )
                        edge.status = 2;

                    cellEdges.edges[ e ] = edge;
                }

                // At this stage, no need to load vertices since they are all fully known
                CellVertices cellVertices { 0, vertices + TotalMaxVertexNb * warp.meta_group_rank() };

                const Real4  si             = context.sites[ i ];
                const float4 fsi            = toFloat( si );
                bool         didCellChanged = false;

                const uint32_t edgeStepNb = ( cellEdges.size / WarpSize ) + 1;
                for ( uint32_t es = 0; es < edgeStepNb; es++ )
                {
                    const uint32_t eid = es * WarpSize + warp.thread_rank();

                    Edge *   edge       = nullptr;
                    uint32_t k          = 0xffffffff;
                    uint32_t l          = 0xffffffff;
                    bool     needSearch = eid < cellEdges.size;
                    if ( needSearch )
                    {
                        edge       = &cellEdges.edges[ eid ];
                        k          = edge->j;
                        l          = edge->k;
                        needSearch = edge->status == 2;
                    }

                    uint32_t edgeType = 0;
                    Real4    edgeVertices[ 2 ];
                    Real4    sk, sl;
                    if ( needSearch )
                    {
                        sk = context.sites[ edge->j ];
                        sl = context.sites[ edge->k ];

                        edgeType = apo::gpu::trisectorVertex( si, sk, sl, edgeVertices[ 0 ], edgeVertices[ 1 ] );
                    }

                    const bool isHyperbola = edgeType == 1;
                    const bool isEllipsis  = edgeType == 2;

                    const float4 v1 = toFloat( edgeVertices[ 0 ] );
                    const float4 v2 = toFloat( edgeVertices[ 1 ] );
                    const float3 c1 = make_float3( v1 );
                    const float3 c2 = make_float3( v2 );

                    float3 n1 = { 0.f, 0.f, 0.f }, n2 = { 0.f, 0.f, 0.f };
                    float3 pp1 = { 0.f, 0.f, 0.f }, pp2 = { 0.f, 0.f, 0.f };

                    float4 boundingSphere = float4 { 0, 0, 0, 0 };
                    if ( needSearch && isEllipsis )
                    {
                        const float3 center = ( c1 + c2 ) * .5f;
                        const float  radius = length( center - c1 ) + v1.w;
                        boundingSphere      = make_float4( center, radius );
                    }
                    else if ( needSearch )
                    {
                        boundingSphere = v1;

                        Real3 nn1, nn2;
                        getTangentPlanes( si, sk, sl, nn1, nn2 );

                        const Real3 p = Real3 { si.x, si.y, si.z } + nn1 * si.w;

                        n1 = toFloat( nn1 );
                        n2 = toFloat( nn2 );

                        const Real3 p1 = Real3 { si.x, si.y, si.z } + nn1 * si.w;
                        const Real3 p2 = Real3 { sk.x, sk.y, sk.z } + nn1 * sk.w;
                        const Real3 p3 = Real3 { sl.x, sl.y, sl.z } + nn1 * sl.w;
                        const Real3 p4 = Real3 { si.x, si.y, si.z } + nn2 * si.w;
                        const Real3 p5 = Real3 { sk.x, sk.y, sk.z } + nn2 * sk.w;

                        boundingSphere = toFloat( sphereFromPoints( p1, p2, p3, p5 ) );

                        pp1 = toFloat( p1 );
                        pp2 = toFloat( p4 );
                    }

                    const bool isInSecurityRadius = apo::lessThan(
                        length( make_float3( fsi ) - make_float3( boundingSphere ) ), knownRadius - boundingSphere.w );
                    if ( eid < cellEdges.size && isInSecurityRadius )
                    {
                        edge->status = 1;
                        needSearch   = false;
                    }

                    uint32_t   stack[ 32 ];
                    uint32_t * stackPtr = stack;
                    uint32_t   nodeId   = 0; // Starting from the root

                    uint32_t foundContributor  = 0xffffffff;
                    bool     continueTraversal = needSearch;
                    while ( true )
                    {
                        while ( continueTraversal )
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

                            const float3 closestLeft  = leftBB.closestPoint( boundingSphere );
                            const float3 closestRight = rightBB.closestPoint( boundingSphere );
                            const float  dLeft        = leftBB.distance( boundingSphere );
                            const float  dRight       = rightBB.distance( boundingSphere );

                            const bool     isLeftClosest = dLeft < dRight;
                            const float    minFarthest   = isLeftClosest ? dRight : dLeft;
                            const float3   farthestPoint = isLeftClosest ? closestRight : closestLeft;
                            const uint32_t farthestChild = isLeftClosest ? right : left;
                            const float    minClosest    = isLeftClosest ? dLeft : dRight;
                            const float3   closestPoint  = isLeftClosest ? closestLeft : closestRight;
                            const uint32_t closestChild  = isLeftClosest ? left : right;

                            bool exploreFarthest = minFarthest < 0.f;
                            if ( isHyperbola && exploreFarthest )
                            {
                                exploreFarthest &= dot( n1, farthestPoint - pp1 ) < 0.f && //
                                                   dot( n2, farthestPoint - pp2 ) < 0.f;
                            }

                            bool exploreClosest = minClosest < 0.f;
                            if ( isHyperbola && exploreClosest )
                            {
                                exploreClosest &= dot( n1, closestPoint - pp1 ) < 0.f && //
                                                  dot( n2, closestPoint - pp2 ) < 0.f;
                            }

                            if ( !exploreFarthest && !exploreClosest )
                                break;

                            nodeId = exploreClosest ? closestChild : farthestChild;
                            if ( exploreClosest && exploreFarthest )
                            {
                                *stackPtr = farthestChild;
                                stackPtr++;
                            }
                        }

                        const bool     isLeafValid = nodeId >= bvh.count - 1 && nodeId < ( 2 * bvh.count - 1 );
                        const uint32_t j           = nodeId - ( bvh.count - 1 );

                        Real4 sj;
                        bool  canContribute = isLeafValid && continueTraversal && j != i && j != edge->j && j != edge->k
                                             && context.status[ j ] != CellStatus::Buried;
                        if ( canContribute )
                        {
                            sj              = context.sites[ j ];
                            const float3 cj = make_float3( toFloat( sj ) );

                            canContribute = apo::gpu::intersect( toFloat( sj ), boundingSphere );

                            if ( isHyperbola )
                                canContribute &= dot( n1, cj - pp1 ) < -sj.w && dot( n2, cj - pp2 ) < -sj.w;
                        }

                        apo::gpu::forEach(
                            warp.ballot( canContribute ),
                            [ & ]( const uint8_t tid )
                            {
                                const uint32_t currentJ = warp.shfl( j, tid );

                                bool           isCurrentKnown = false;
                                const uint32_t edgeSteps      = ( cellEdges.size / WarpSize ) + 1;
                                for ( uint8_t ess = 0; ess < edgeSteps && !isCurrentKnown; ess++ )
                                {
                                    const uint32_t eeid = warp.thread_rank() + WarpSize * ess;

                                    if ( eeid < cellEdges.size )
                                    {
                                        const Edge & currentEdge = cellEdges.edges[ eeid ];
                                        isCurrentKnown           = eeid < cellEdges.size
                                                         && ( currentEdge.j == currentJ || currentEdge.k == currentJ );
                                    }
                                    isCurrentKnown = warp.any( isCurrentKnown );
                                }

                                if ( warp.thread_rank() == tid )
                                    canContribute = !isCurrentKnown;
                            } );

                        uint8_t mask = 0;
                        Real4   x[ 2 ];
                        if ( canContribute )
                        {
                            mask = apo::gpu::quadrisector( si, sj, sk, sl, x[ 0 ], x[ 1 ] );

                            if ( mask != 3 )
                                mask = 0;
                        }

                        uint8_t valid = mask; // Two booleans
                        apo::gpu::forEach(
                            warp.ballot( mask ),
                            [ & ]( const uint8_t tid )
                            {
                                uint8_t        currentMask = warp.shfl( mask, tid );
                                const uint32_t currentK    = warp.shfl( k, tid );
                                const uint32_t currentL    = warp.shfl( l, tid );
                                const uint32_t currentJ    = warp.shfl( j, tid );

                                while ( currentMask != 0 )
                                {
                                    const uint8_t nv = __ffs( currentMask ) - 1;
                                    currentMask &= ( ~( 1u << ( nv ) ) );

                                    const Real4    currentVertex = warp.shfl( x[ nv ], tid );
                                    bool           isValid       = true;
                                    const uint32_t edgeSteps     = ( cellEdges.size / WarpSize ) + 1;
                                    for ( uint8_t ess = 0; ess < edgeSteps && isValid; ess++ )
                                    {
                                        const uint32_t eeid        = warp.thread_rank() + WarpSize * ess;
                                        const Edge &   currentEdge = cellEdges.edges[ eeid ];
                                        if ( eeid < cellEdges.size && currentEdge.j != currentK
                                             && currentEdge.j != currentL )
                                            isValid &= !intersect( context.sites[ currentEdge.j ], currentVertex );

                                        if ( eeid < cellEdges.size && currentEdge.k != currentK
                                             && currentEdge.k != currentL )
                                            isValid &= !intersect( context.sites[ currentEdge.k ], currentVertex );

                                        isValid = warp.all( isValid );
                                    }

                                    if ( warp.thread_rank() == tid )
                                        valid &= ~( !isValid << nv );
                                }
                            } );

                        if ( valid )
                        {
                            foundContributor  = j;
                            continueTraversal = false;
                        }

                        while ( continueTraversal )
                        {
                            if ( stackPtr == stack )
                            {
                                continueTraversal = false;
                                break;
                            }

                            --stackPtr;

                            nodeId = *stackPtr;
                            break;
                        }

                        const uint32_t continueSearch = warp.ballot( continueTraversal );
                        if ( !continueSearch )
                            break;
                    }

                    if ( needSearch && foundContributor == 0xffffffff )
                        edge->status = 1;

                    bool cellChanged = false;
                    apo::gpu::forEach( warp.ballot( foundContributor != 0xffffffff ),
                                       [ & ]( const uint8_t tid )
                                       {
                                           const uint32_t currentJ       = warp.shfl( foundContributor, tid );
                                           bool           isCurrentKnown = false;

                                           // Test again since the data structure may have been modified
                                           const uint32_t edgeSteps = ( cellEdges.size / WarpSize ) + 1;
                                           for ( uint8_t ess = 0; ess < edgeSteps && !isCurrentKnown; ess++ )
                                           {
                                               const uint32_t eeid        = warp.thread_rank() + WarpSize * ess;
                                               const Edge &   currentEdge = cellEdges.edges[ eeid ];
                                               isCurrentKnown             = warp.any(
                                                   eeid < cellEdges.size
                                                   && ( currentEdge.j == currentJ || currentEdge.k == currentJ ) );
                                           }

                                           if ( isCurrentKnown )
                                               return;

                                           const Real4 sj = context.sites[ currentJ ];
                                           cellChanged = updateCell<TotalMaxVertexNb, TotalMaxEdgeNb, MaxNewEdgeNb>( //
                                               warp,
                                               context,
                                               i,
                                               si,
                                               cellVertices,
                                               cellEdges,
                                               currentJ,
                                               sj,
                                               tempStorage.newEdgeSort );

                                           didCellChanged |= cellChanged;

                                           // If cell changed, we can't ensure that new edges are not positioned at the
                                           // start of the buffer Then, we restart validation.
                                           if ( cellChanged )
                                               es = 0;
                                       } );
                }

                const uint32_t baseVertexNb = context.vertexNb[ i ];
                if ( threadIdx.x % warpSize == 0 )
                {
                    context.status[ i ]   = !didCellChanged;
                    context.edgeNb[ i ]   = cellEdges.size;
                    context.vertexNb[ i ] = baseVertexNb + cellVertices.size;

                    if ( baseVertexNb + cellVertices.size > TotalMaxVertexNb )
                        printf( "Error on cell %d cellEdges.size = %d\n", i, cellEdges.size );
                }

                for ( uint32_t e = warp.thread_rank(); e < cellEdges.size; e += warpSize )
                    context.edges[ i * TotalMaxEdgeNb + e ] = cellEdges.edges[ e ];

                for ( uint32_t v = warp.thread_rank(); v < cellVertices.size; v += warpSize )
                    context.vertices[ i * TotalMaxVertexNb + baseVertexNb + v ] = cellVertices.vertices[ v ];
            }
        }

        inline __global__ void globalBisectorValidation( ContextCellOriented context,
                                                         LBVH::View          bvh,
                                                         const float * const knownRadii )
        {
            constexpr uint32_t MaxBisectorPerThread = MaxBisectorNb / WarpSize;
            using WarpMergeSortEdgeValidationT      = cub::WarpMergeSort<uint32_t, MaxBisectorPerThread, WarpSize>;
            union TempStorage
            {
                WarpMergeSortNewEdgeT<MaxNewEdgeNb>::TempStorage newEdgeSort;
                WarpMergeSortEdgeValidationT::TempStorage        bisectorSort;
            };

            constexpr uint16_t     WarpPerBlock = BlockSize / WarpSize;
            __shared__ TempStorage tempStorages[ WarpPerBlock ];
            __shared__ Edge        edges[ TotalMaxEdgeNb * WarpPerBlock ];
            __shared__ uint32_t    bisectors[ MaxBisectorNb * WarpPerBlock ];
            __shared__ Vertex      vertices[ TotalMaxVertexNb * WarpPerBlock ];

            const auto     block  = cg::this_thread_block();
            const auto     warp   = cg::tiled_partition<WarpSize>( block );
            const uint32_t warpId = threadIdx.x / WarpSize;

            const uint32_t start = blockIdx.x * WarpPerBlock;
            const uint32_t end   = apo::min( ( blockIdx.x + 1 ) * WarpPerBlock, context.siteNb );

            TempStorage & tempStorage = tempStorages[ warp.meta_group_rank() ];
            for ( uint32_t i = start + warpId; i < end; i += WarpPerBlock )
            {
                const uint8_t status = context.status[ i ];
                if ( status == CellStatus::Buried )
                    continue;

                const float knownRadius = knownRadii[ i ];

                // Load edges
                const uint32_t totalEdgeNb = context.edgeNb[ i ];
                CellEdges      cellEdges   = {
                    totalEdgeNb,
                    edges + TotalMaxEdgeNb * warp.meta_group_rank(),
                };

                for ( uint32_t e = warp.thread_rank(); e < totalEdgeNb; e += warpSize )
                    cellEdges.edges[ e ] = context.edges[ i * TotalMaxEdgeNb + e ];

                uint32_t * cellBisectors = bisectors + warp.meta_group_rank() * MaxBisectorNb;
                uint32_t   bisectorNb    = 0;
                {
                    uint32_t localBisectors[ MaxBisectorPerThread ];
                    for ( uint32_t e = 0; e < MaxBisectorPerThread; e++ )
                        localBisectors[ e ] = 0xffffffff;

                    uint32_t localBisectorNb = 0;
                    for ( uint32_t eid = warp.thread_rank(); eid < cellEdges.size; eid += warpSize )
                    {
                        const apo::gpu::Edge edge = cellEdges.edges[ eid ];
                        if ( edge.j < context.siteNb )
                            localBisectors[ localBisectorNb++ ] = edge.j;
                        if ( edge.k < context.siteNb )
                            localBisectors[ localBisectorNb++ ] = edge.k;

                        if ( localBisectorNb >= MaxBisectorPerThread )
                            printf( "Error: Too much bisectors. Increase MaxBisectorPerThread\n" );
                    }

                    WarpMergeSortEdgeValidationT( tempStorage.bisectorSort )
                        .Sort( localBisectors, [] __device__( const uint32_t a, const uint32_t b ) { return a < b; } );

                    uint32_t isFirstBisector = 0;
                    for ( uint32_t b = 1; b < MaxBisectorPerThread; b++ )
                    {
                        const uint32_t previous = localBisectors[ b - 1 ];
                        const uint32_t current  = localBisectors[ b ];
                        const uint32_t flag     = ( current != previous ) && ( current != 0xffffffff );
                        isFirstBisector |= ( flag && ( current > i ) ) << b;
                    }

                    const uint32_t previous = warp.shfl_up( localBisectors[ MaxBisectorPerThread - 1 ], 1 );
                    const uint32_t first    = localBisectors[ 0 ];
                    if ( warp.thread_rank() != 0 )
                    {
                        const uint32_t flag = ( first != 0xffffffff ) && ( first != previous );
                        isFirstBisector |= flag && ( first > i );
                    }
                    else
                    {
                        const uint32_t flag = ( first != 0xffffffff );
                        isFirstBisector |= flag && ( first > i );
                    }

                    const uint32_t localUniqueBisectorNb = __popc( isFirstBisector );
                    const uint32_t offset
                        = apo::gpu::warpExclusiveScan<uint32_t, WarpSize>( warp, localUniqueBisectorNb );
                    bisectorNb = warp.shfl( offset + localUniqueBisectorNb, WarpSize - 1 );
                    if ( bisectorNb >= MaxBisectorNb && warp.thread_rank() == 0 )
                        printf( "%d - Increase MaxBisectorNb !\n", i );

                    uint32_t localOffset = 0;
                    for ( uint32_t b = 0; b < MaxBisectorPerThread; b++ )
                    {
                        const bool isFirst = ( isFirstBisector >> b ) & 1;
                        if ( !isFirst )
                            continue;

                        cellBisectors[ offset + localOffset ] = localBisectors[ b ];
                        localOffset++;

                        if ( offset + localOffset >= MaxBisectorNb )
                            printf( "%d - Increase MaxBisectorNb !\n", i );
                    }
                }

                // At this stage, no need to load vertices since they are all fully known
                CellVertices cellVertices { 0, vertices + TotalMaxVertexNb * warp.meta_group_rank() };

                const Real4  si  = context.sites[ i ];
                const float3 fci = make_float3( si.x, si.y, si.z );
                const float  fri = static_cast<float>( si.w );

                bool           didCellChanged = false;
                const uint32_t bisectorStepNb = ( bisectorNb / WarpSize ) + 1;
                for ( uint32_t bs = 0; bs < bisectorStepNb; bs++ )
                {
                    const uint32_t bid = bs * WarpSize + warp.thread_rank();

                    uint32_t bisector   = 0xffffffff;
                    bool     needSearch = bid < bisectorNb;
                    if ( needSearch )
                    {
                        bisector   = cellBisectors[ bid ];
                        needSearch = bisector != 0xffffffff;
                    }

                    Real4  sk  = Real4 { 0, 0, 0, 0 };
                    float3 fck = float3 { 0, 0, 0 };
                    float  frk = 0.f;
                    if ( needSearch )
                    {
                        sk  = context.sites[ bisector ];
                        fck = make_float3( sk.x, sk.y, sk.z );
                        frk = static_cast<float>( sk.w );

                        const float distance = length( fci - fck ) + fri + frk;
                        if ( apo::lessThan( distance, knownRadius ) )
                        {
                            cellBisectors[ bid ] = 0xffffffff;
                            needSearch           = false;
                        }
                    }

                    float  cylinderRadius = 0.f;
                    float4 boundingSphere = float4 { 0, 0, 0, 0 };
                    if ( needSearch )
                    {
                        const float3 center = ( fci + fck ) * Real( .5 );
                        const float  radius = length( center - fci ) + apo::max( fri, frk );
                        boundingSphere      = float4 { center.x, center.y, center.z, radius };

                        cylinderRadius = apo::max( fri, frk );
                    }

                    uint32_t   stack[ 32 ];
                    uint32_t * stackPtr = stack;
                    uint32_t   nodeId   = 0; // Starting from the root

                    uint32_t foundContributor  = 0xffffffff;
                    bool     continueTraversal = needSearch;
                    while ( true )
                    {
                        while ( continueTraversal )
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

                            const float3 closestLeft  = leftBB.closestPoint( boundingSphere );
                            const float3 closestRight = rightBB.closestPoint( boundingSphere );
                            const float  dLeft        = leftBB.distance( boundingSphere );
                            const float  dRight       = rightBB.distance( boundingSphere );

                            const bool     isLeftClosest = dLeft < dRight;
                            const float    minFarthest   = isLeftClosest ? dRight : dLeft;
                            const float3   farthestPoint = isLeftClosest ? closestRight : closestLeft;
                            const uint32_t farthestChild = isLeftClosest ? right : left;
                            const float    minClosest    = isLeftClosest ? dLeft : dRight;
                            const float3   closestPoint  = isLeftClosest ? closestLeft : closestRight;
                            const uint32_t closestChild  = isLeftClosest ? left : right;

                            const bool exploreFarthest = minFarthest < 0.f;
                            const bool exploreClosest  = minClosest < 0.f;

                            if ( !exploreFarthest && !exploreClosest )
                                break;

                            nodeId = exploreClosest ? closestChild : farthestChild;
                            if ( exploreClosest && exploreFarthest )
                            {
                                *stackPtr = farthestChild;
                                stackPtr++;
                            }
                        }

                        const bool     isLeafValid = nodeId >= bvh.count - 1 && nodeId < ( 2 * bvh.count - 1 );
                        const uint32_t j           = nodeId - ( bvh.count - 1 );

                        Real4 sj;
                        bool  canContribute = isLeafValid && continueTraversal && j != i && j != bisector;
                        if ( canContribute )
                        {
                            sj               = context.sites[ j ];
                            const float4 fsj = apo::gpu::toFloat( sj );
                            const float3 fcj = make_float3( fsj );

                            canContribute
                                = boundingSphere.w >= fsj.w
                                  && ( length( make_float3( boundingSphere ) - fcj ) <= ( boundingSphere.w - fsj.w ) );
                            if ( canContribute )
                            {
                                const float3 itoj = fcj - fci;
                                const float  v    = dot( itoj, normalize( fck - fci ) );
                                const float  d    = ::sqrtf( dot( itoj, itoj ) - v * v );
                                canContribute &= ( d + fsj.w ) < cylinderRadius;
                            }
                        }

                        apo::gpu::forEach( warp.ballot( canContribute ),
                                           [ & ]( const uint8_t tid )
                                           {
                                               const uint32_t currentJ = warp.shfl( j, tid );

                                               bool           isCurrentKnown = false;
                                               const uint32_t edgeSteps      = ( cellEdges.size / WarpSize ) + 1;
                                               for ( uint8_t ess = 0; ess < edgeSteps && !isCurrentKnown; ess++ )
                                               {
                                                   const uint32_t eeid        = warp.thread_rank() + WarpSize * ess;
                                                   const Edge &   currentEdge = cellEdges.edges[ eeid ];
                                                   isCurrentKnown             = warp.any(
                                                       eeid < cellEdges.size
                                                       && ( currentEdge.j == currentJ || currentEdge.k == currentJ ) );
                                               }

                                               if ( warp.thread_rank() == tid )
                                                   canContribute = !isCurrentKnown;
                                           } );

                        uint8_t vertexNb = 0;
                        Real4   vertex;
                        if ( canContribute )
                        {
                            Real4 x[ 2 ];
                            vertexNb = apo::gpu::trisectorVertex( si, sj, sk, x[ 0 ], x[ 1 ] );
                            vertex   = x[ 1 ];

                            canContribute = vertexNb == 2;
                        }

                        apo::gpu::forEach(
                            warp.ballot( canContribute ),
                            [ & ]( const uint8_t tid )
                            {
                                const uint32_t currentK = warp.shfl( bisector, tid );
                                const uint32_t currentJ = warp.shfl( j, tid );

                                const Real4    currentVertex = warp.shfl( vertex, tid );
                                bool           isValid       = true;
                                const uint32_t edgeSteps     = ( cellEdges.size / WarpSize ) + 1;
                                for ( uint8_t ess = 0; ess < edgeSteps && isValid; ess++ )
                                {
                                    const uint32_t eeid        = warp.thread_rank() + WarpSize * ess;
                                    const Edge &   currentEdge = cellEdges.edges[ eeid ];
                                    if ( eeid < cellEdges.size && currentEdge.j != currentK )
                                        isValid &= !intersect( context.sites[ currentEdge.j ], currentVertex );

                                    if ( eeid < cellEdges.size && currentEdge.k != currentK )
                                        isValid &= !intersect( context.sites[ currentEdge.k ], currentVertex );

                                    isValid = warp.all( isValid );
                                }

                                if ( warp.thread_rank() == tid )
                                    canContribute &= isValid;
                            } );

                        if ( canContribute )
                        {
                            foundContributor  = j;
                            continueTraversal = false;
                        }

                        while ( continueTraversal )
                        {
                            if ( stackPtr == stack )
                            {
                                continueTraversal = false;
                                break;
                            }

                            --stackPtr;

                            nodeId = *stackPtr;
                            break;
                        }

                        const uint32_t continueSearch = warp.ballot( continueTraversal );
                        if ( !continueSearch )
                            break;
                    }

                    if ( needSearch && foundContributor == 0xffffffff )
                        cellBisectors[ bid ] = 0xffffffff;

                    bool cellChanged = false;
                    apo::gpu::forEach(
                        warp.ballot( foundContributor != 0xffffffff ),
                        [ & ]( const uint8_t tid )
                        {
                            const uint32_t currentJ       = warp.shfl( foundContributor, tid );
                            bool           isCurrentKnown = false;

                            // Test again since the data structure may have been modified
                            const uint32_t edgeSteps = ( cellEdges.size / WarpSize ) + 1;
                            for ( uint8_t ess = 0; ess < edgeSteps && !isCurrentKnown; ess++ )
                            {
                                const uint32_t eeid        = warp.thread_rank() + WarpSize * ess;
                                const Edge &   currentEdge = cellEdges.edges[ eeid ];
                                isCurrentKnown
                                    = warp.any( eeid < cellEdges.size
                                                && ( currentEdge.j == currentJ || currentEdge.k == currentJ ) );
                            }

                            if ( isCurrentKnown )
                                return;

                            const Real4 sj = context.sites[ currentJ ];
                            cellChanged    = updateCell<TotalMaxVertexNb, TotalMaxEdgeNb, MaxNewEdgeNb>(
                                warp, context, i, si, cellVertices, cellEdges, currentJ, sj, tempStorage.newEdgeSort );

                            didCellChanged |= cellChanged;

                            // If cell changed, we can't ensure that new edges are not positioned at the
                            // start of the buffer Then, we restart validation.
                            if ( cellChanged )
                                bs = 0;
                        } );
                }

                const uint32_t baseVertexNb = context.vertexNb[ i ];
                if ( threadIdx.x % warpSize == 0 )
                {
                    context.status[ i ]   = !didCellChanged;
                    context.edgeNb[ i ]   = cellEdges.size;
                    context.vertexNb[ i ] = baseVertexNb + cellVertices.size;

                    if ( baseVertexNb + cellVertices.size > TotalMaxVertexNb )
                        printf( "Error on cell %d cellEdges.size = %d\n", i, cellEdges.size );
                }

                for ( uint32_t e = warp.thread_rank(); e < cellEdges.size; e += warpSize )
                    context.edges[ i * TotalMaxEdgeNb + e ] = cellEdges.edges[ e ];

                for ( uint32_t v = warp.thread_rank(); v < cellVertices.size; v += warpSize )
                    context.vertices[ i * TotalMaxVertexNb + baseVertexNb + v ] = cellVertices.vertices[ v ];
            }
        }

        inline __global__ void getWriteableVertices( ContextCellOriented context, uint32_t * const startWriteIndices )
        {
            const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if ( id >= context.siteNb )
                return;

            const uint32_t vertexNb = context.vertexNb[ id ];
            const uint8_t  status   = context.status[ id ];
            if ( status == CellStatus::Buried )
            {
                startWriteIndices[ id ] = 0;
                return;
            }

            uint32_t writeableVertexNb = 0;
            for ( uint32_t v = 0; v < vertexNb; v++ )
            {
                const Vertex vertex = context.vertices[ id * TotalMaxVertexNb + v ];
                if ( id > vertex.j || vertex.j >= context.siteNb || vertex.k >= context.siteNb
                     || vertex.l >= context.siteNb )
                    continue;

                writeableVertexNb++;
            }

            startWriteIndices[ id ] = writeableVertexNb;
        }

        inline __global__ void saveVertices( ContextCellOriented context,
                                             Real4 * __restrict__ output,
                                             uint32_t * __restrict__ outputIds,
                                             uint32_t * __restrict__ startWriteIndices,
                                             const LBVH::View bvh )
        {
            const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if ( id >= context.siteNb )
                return;

            const uint8_t status = context.status[ id ];
            if ( status == CellStatus::Buried )
                return;

            const uint32_t start    = startWriteIndices[ id ];
            const uint32_t vertexNb = context.vertexNb[ id ];

            uint32_t currentWritingIndex = 0;
            for ( uint32_t v = 0; v < vertexNb; v++ )
            {
                if ( DebugCellConstruction && id != DebugId )
                    continue;

                const Vertex vertex = context.vertices[ id * TotalMaxVertexNb + v ];
                if ( id > vertex.j || vertex.j >= context.siteNb || vertex.k >= context.siteNb
                     || vertex.l >= context.siteNb )
                    continue;

                const Real4 si = context.sites[ id ];
                const Real4 sj = context.sites[ vertex.j ];
                const Real4 sk = context.sites[ vertex.k ];
                const Real4 sl = context.sites[ vertex.l ];

                Real4 x[ 2 ];
                apo::gpu::quadrisector( si, sj, sk, sl, x[ 0 ], x[ 1 ] );
                output[ start + currentWritingIndex ] = x[ vertex.type ];

                outputIds[ ( start + currentWritingIndex ) * 4 + 0 ] = bvh.indices[ id ];
                outputIds[ ( start + currentWritingIndex ) * 4 + 1 ] = bvh.indices[ vertex.j ];
                outputIds[ ( start + currentWritingIndex ) * 4 + 2 ] = bvh.indices[ vertex.k ];
                outputIds[ ( start + currentWritingIndex ) * 4 + 3 ] = bvh.indices[ vertex.l ];

                currentWritingIndex++;
            }
        }

        inline __global__ void getWriteableClosedEdges( ContextCellOriented context,
                                                        uint32_t * const    startWriteIndices )
        {
            const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if ( id >= context.siteNb )
                return;

            const uint32_t edgeNb = context.edgeNb[ id ];
            const uint8_t  status = context.status[ id ];
            if ( status == CellStatus::Buried )
            {
                startWriteIndices[ id ] = 0;
                return;
            }

            uint32_t writeableEdgeNb = 0;
            for ( uint32_t e = 0; e < edgeNb; e++ )
            {
                const Edge edge = context.edges[ id * TotalMaxEdgeNb + e ];
                if ( edge.vertexNb != 0xffffffff || id > edge.j || edge.j >= context.siteNb
                     || edge.k >= context.siteNb )
                    continue;

                writeableEdgeNb++;
            }

            startWriteIndices[ id ] = writeableEdgeNb;
        }

        inline __global__ void saveClosedEdges( ContextCellOriented context,
                                                Real4 * __restrict__ output,
                                                uint32_t * __restrict__ outputIds,
                                                uint32_t * __restrict__ startWriteIndices,
                                                const LBVH::View bvh )
        {
            const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if ( id >= context.siteNb )
                return;

            const uint8_t status = context.status[ id ];
            if ( status == CellStatus::Buried )
                return;

            const uint32_t start  = startWriteIndices[ id ];
            const uint32_t edgeNb = context.edgeNb[ id ];

            const Real4 si = context.sites[ id ];

            uint32_t currentWritingIndex = 0;
            for ( uint32_t e = 0; e < edgeNb; e++ )
            {
                const Edge edge = context.edges[ id * TotalMaxEdgeNb + e ];

                if ( edge.vertexNb != 0xffffffff || id > edge.j || edge.j >= context.siteNb
                     || edge.k >= context.siteNb )
                    continue;

                const Real4 sj = context.sites[ edge.j ];
                const Real4 sk = context.sites[ edge.k ];

                Real4   x[ 2 ];
                uint8_t count = apo::gpu::trisectorVertex( si, sj, sk, x[ 0 ], x[ 1 ] );

                if ( x[ 1 ].w < x[ 0 ].w )
                    x[ 0 ] = x[ 1 ];

                output[ start + currentWritingIndex ] = x[ 0 ];

                outputIds[ ( start + currentWritingIndex ) * 4 + 0 ] = bvh.indices[ id ];
                outputIds[ ( start + currentWritingIndex ) * 4 + 1 ] = bvh.indices[ edge.j ];
                outputIds[ ( start + currentWritingIndex ) * 4 + 2 ] = bvh.indices[ edge.k ];

                currentWritingIndex++;
            }
        }

        constexpr std::size_t nextPowerOfTwoValue( const std::size_t baseNumber )
        {
            std::size_t i = 1;
            while ( baseNumber > i )
                i <<= 1;
            return i;
        }

        constexpr std::size_t nextPowerOfTwoExponent( std::size_t baseNumber )
        {
            uint32_t exponent = 0;
            while ( baseNumber >>= 1 )
            {
                exponent++;
            }
            return exponent;
        }
    } // namespace impl

    template<uint32_t K>
    AlgorithmGPU<K>::AlgorithmGPU( ConstSpan<apo::Real> sites, uint32_t knnValidationStepNb ) :
        Algorithm( sites ), knnValidationStep( knnValidationStepNb )
    {
    }

    template<uint32_t K>
    void AlgorithmGPU<K>::build()
    {
        sample = Sample {};

        ContextCellOriented context {};
        context.siteNb = m_siteNb;

        dSites            = DeviceBuffer { ( m_siteNb + ArtificialSiteNb ) * sizeof( Real4 ), false };
        context.sites     = dSites.get<Real4>();
        context.minRadius = m_minRadius;
        context.maxRadius = m_maxRadius;

        // Storage for data structure
        dStatus        = { m_siteNb * sizeof( uint8_t ), true };
        context.status = dStatus.get<uint8_t>();

        dVertices        = DeviceBuffer { sizeof( Vertex ) * impl::TotalMaxVertexNb * m_siteNb, false };
        dVerticesNb      = DeviceBuffer { sizeof( uint32_t ) * m_siteNb, true };
        context.vertexNb = dVerticesNb.get<uint32_t>();
        context.vertices = dVertices.get<Vertex>();

        dEdges         = DeviceBuffer { sizeof( Edge ) * impl::TotalMaxEdgeNb * m_siteNb, false };
        dEdgesNb       = DeviceBuffer { sizeof( uint32_t ) * m_siteNb, true };
        context.edges  = dEdges.get<Edge>();
        context.edgeNb = dEdgesNb.get<uint32_t>();

        const Real3 bbMin = Real3 { m_minX, m_minY, m_minZ };
        const Real3 bbMax = Real3 { m_maxX, m_maxY, m_maxZ };

        // Transfer sites to device and append artificial sites at the end
        mmemcpy<MemcpyType::HostToDevice>( dSites.get<Real>(), m_sites.ptr, m_sites.size );
        {
            const Real3 centroidCenter = ( bbMin + bbMax ) / apo::Real( 2 );
            const Real4 centroid       = { centroidCenter.x, centroidCenter.y, centroidCenter.z, Real( 0 ) };
            const Real  aabbRadius     = length( centroidCenter - Real3 { m_maxX, m_maxY, m_maxZ } );

            // Set a threshold to avoid numerical errors
            constexpr Real Threshold = Real( 1e-1 );
            const Real     radius    = aabbRadius * ( std::sqrt( Real( 3. ) ) + Threshold );

            const std::vector<Real4> artificialSites = {
                centroid + Real4 { radius, Real( 0 ), Real( 0 ), context.minRadius },
                centroid + Real4 { -radius, Real( 0 ), Real( 0 ), context.minRadius },
                centroid + Real4 { Real( 0 ), radius, Real( 0 ), context.minRadius },
                centroid + Real4 { Real( 0 ), -radius, Real( 0 ), context.minRadius },
                centroid + Real4 { Real( 0 ), Real( 0 ), radius, context.minRadius },
                centroid + Real4 { Real( 0 ), Real( 0 ), -radius, context.minRadius },
            };

            apo::joggle( { (Real *)artificialSites.data(), ArtificialSiteNb * 4 }, 1e-1 );

            mmemcpy<MemcpyType::HostToDevice>(
                dSites.get<Real4>() + context.siteNb, artificialSites.data(), ArtificialSiteNb );
        }

        // Load artificial sites to constant memory
        constexpr std::size_t ArtificialVerticesDataCount = ArtificialVerticesNb * 4;
        constexpr std::size_t ArtificialEdgesDataCount    = ArtificialEdgesNb * 2;
        DeviceBuffer          artificialData
            = DeviceBuffer::Typed<uint32_t>( ArtificialVerticesDataCount + ArtificialEdgesDataCount );
        {
            const uint32_t artificialVerticesIndices[ ArtificialVerticesNb * 4 ] = {
                context.siteNb + 0, context.siteNb + 2, context.siteNb + 4, 0, //
                context.siteNb + 0, context.siteNb + 2, context.siteNb + 5, 0, //
                context.siteNb + 0, context.siteNb + 3, context.siteNb + 4, 0, //
                context.siteNb + 0, context.siteNb + 3, context.siteNb + 5, 0, //
                context.siteNb + 1, context.siteNb + 2, context.siteNb + 4, 0, //
                context.siteNb + 1, context.siteNb + 2, context.siteNb + 5, 0, //
                context.siteNb + 1, context.siteNb + 3, context.siteNb + 4, 0, //
                context.siteNb + 1, context.siteNb + 3, context.siteNb + 5, 0, //
            };

            apo::gpu::mmemcpy<MemcpyType::HostToDevice>(
                artificialData.get<uint32_t>(), artificialVerticesIndices, ArtificialVerticesDataCount );

            const uint32_t artificialEdgesIndices[ ArtificialEdgesNb * 2 ] = {
                context.siteNb + 0, context.siteNb + 2, //
                context.siteNb + 2, context.siteNb + 4, //
                context.siteNb + 1, context.siteNb + 2, //
                context.siteNb + 2, context.siteNb + 5, //
                context.siteNb + 1, context.siteNb + 4, //
                context.siteNb + 1, context.siteNb + 5, //
                context.siteNb + 0, context.siteNb + 5, //
                context.siteNb + 0, context.siteNb + 4, //
                context.siteNb + 1, context.siteNb + 3, //
                context.siteNb + 3, context.siteNb + 5, //
                context.siteNb + 0, context.siteNb + 3, //
                context.siteNb + 3, context.siteNb + 4, //
            };

            apo::gpu::mmemcpy<MemcpyType::HostToDevice>( //
                artificialData.get<uint32_t>() + ArtificialVerticesDataCount,
                artificialEdgesIndices,
                ArtificialEdgesDataCount );
        }

        // BVH construction
        bvh                    = LBVH {};
        Aabbf aabb             = { make_float3( m_minX, m_minY, m_minZ ), make_float3( m_maxX, m_maxY, m_maxZ ) };
        sample.bvhConstruction = timer_ms( [ & ]() { bvh.build( context.siteNb, aabb, context.sites ); } );

        // Actual computation
        // Initialize data structure
        auto [ gridDim, blockDim ] = KernelConfig::From( context.siteNb, impl::KnnBlockSize );
        sample.initialization      = timer_ms(
            [ &, gridDim = gridDim, blockDim = blockDim ]()
            {
                initializeDataStructure<impl::TotalMaxVertexNb, impl::TotalMaxEdgeNb>
                    <<<gridDim, blockDim>>>( context,
                                             artificialData.get<uint32_t>(),
                                             artificialData.get<uint32_t>() + ArtificialVerticesDataCount );
            } );
        apo::gpu::cudaCheck( "initializeDataStructure" );

        DeviceBuffer dKnns = DeviceBuffer( m_siteNb * K * ( sizeof( uint32_t ) + sizeof( Real ) ), false );
        cudaCheck( cudaMemset( dKnns.get<uint32_t>(), 0xff, sizeof( uint32_t ) * m_siteNb * K ) );

        DeviceBuffer validatedEdgesNb    = DeviceBuffer::Typed<uint32_t>( context.siteNb, true );
        DeviceBuffer validatedVerticesNb = DeviceBuffer::Typed<uint32_t>( context.siteNb, true );

        std::vector<float> initRadii  = std::vector<float>( context.siteNb, std::numeric_limits<float>::lowest() );
        DeviceBuffer       knownRadii = DeviceBuffer::Typed<float>( context.siteNb, false );
        mmemcpy<MemcpyType::HostToDevice>( knownRadii.get<float>(), initRadii.data(), context.siteNb );

        for ( uint32_t s = 0; s < knnValidationStep; s++ )
        {
            sample.knnSearch += timer_ms( [ &, gridDim = gridDim, blockDim = blockDim ]() { //
                impl::getKnns<K><<<gridDim, blockDim>>>( s, context, bvh.getDeviceView(), dKnns.get<uint32_t>() );
            } );
            apo::gpu::cudaCheck( "getKnns" );

            auto [ cellGridDim, cellBlockDim ] = KernelConfig::From( context.siteNb * WarpSize, impl::BlockSize );
            sample.knnConstruction += timer_ms(
                [ &, cellGridDim = cellGridDim, cellBlockDim = cellBlockDim ]() {
                    impl::getCells<<<cellGridDim, cellBlockDim>>>(
                        context, K, dKnns.get<uint32_t>(), knownRadii.get<float>() );
                } );
            apo::gpu::cudaCheck( "getCells" );
        }

        uint32_t taskNb = thrust::count(
            thrust::device, context.status, context.status + context.siteNb, CellStatus::InValidation );

        constexpr std::size_t MaxIterationNb = 10;
        for ( std::size_t iteration = 0; iteration < MaxIterationNb && taskNb != 0; iteration++ )
        {
            DeviceBuffer tasks = DeviceBuffer::Typed<uint32_t>( taskNb );
            thrust::copy_if( thrust::device,
                             thrust::make_counting_iterator<uint32_t>( 0 ),
                             thrust::make_counting_iterator<uint32_t>( context.siteNb ),
                             context.status,
                             tasks.get<uint32_t>(),
                             thrust::placeholders::_1 == 0 );

            auto [ cellGridDim, cellBlockDim ] = KernelConfig::From( taskNb * WarpSize, impl::BlockSize );
            sample.vertexValidation += timer_ms(
                [ &, cellGridDim = cellGridDim, cellBlockDim = cellBlockDim ]()
                {
                    impl::globalVertexValidation<<<cellGridDim, cellBlockDim>>>(
                        context, bvh.getDeviceView(), taskNb, tasks.get<uint32_t>(), knownRadii.get<float>() );
                } );
            apo::gpu::cudaCheck( "globalVertexValidation" );

            sample.edgeValidation += timer_ms(
                [ &, cellGridDim = cellGridDim, cellBlockDim = cellBlockDim ]()
                {
                    impl::globalEdgeValidation<<<cellGridDim, cellBlockDim>>>(
                        context, bvh.getDeviceView(), taskNb, tasks.get<uint32_t>(), knownRadii.get<float>() );
                } );
            apo::gpu::cudaCheck( "globalEdgeValidation" );

            taskNb = thrust::count( thrust::device, context.status, context.status + context.siteNb, uint8_t( 0 ) );
        }

        auto [ cellGridDim, cellBlockDim ] = KernelConfig::From( context.siteNb * WarpSize, impl::BlockSize );

        sample.bisectorValidation = timer_ms(
            [ &, cellGridDim = cellGridDim, cellBlockDim = cellBlockDim ]()
            {
                impl::globalBisectorValidation<<<cellGridDim, cellBlockDim>>>(
                    context, bvh.getDeviceView(), knownRadii.get<float>() );
            } );
        apo::gpu::cudaCheck( "globalBisectorValidation" );
    }

    template<uint32_t K>
    VertexDiagram AlgorithmGPU<K>::toHost( bool doValidation )
    {
        ContextCellOriented context {};
        context.sites     = dSites.get<Real4>();
        context.siteNb    = m_siteNb;
        context.minRadius = m_minRadius;
        context.maxRadius = m_maxRadius;

        // Storage for data structure
        context.status   = dStatus.get<uint8_t>();
        context.vertexNb = dVerticesNb.get<uint32_t>();
        context.vertices = dVertices.get<Vertex>();
        context.edges    = dEdges.get<Edge>();
        context.edgeNb   = dEdgesNb.get<uint32_t>();

        //
        DeviceBuffer writeIndices  = DeviceBuffer::Typed<uint32_t>( m_siteNb );
        auto [ gridDim, blockDim ] = KernelConfig::From( context.siteNb, impl::BlockSize );
        impl::getWriteableVertices<<<gridDim, blockDim>>>( context, writeIndices.get<uint32_t>() );
        apo::gpu::cudaCheck( "getWriteableVertices" );

        uint32_t lastVertexNb = 0;
        mmemcpy<MemcpyType::DeviceToHost>( &lastVertexNb, writeIndices.get<uint32_t>() + m_siteNb - 1, 1 );

        thrust::exclusive_scan( thrust::device,
                                writeIndices.get<uint32_t>(),
                                writeIndices.get<uint32_t>() + m_siteNb,
                                writeIndices.get<uint32_t>() );

        uint32_t lastWritingIndex = 0;
        mmemcpy<MemcpyType::DeviceToHost>( &lastWritingIndex, writeIndices.get<uint32_t>() + m_siteNb - 1, 1 );

        const uint32_t vertexNb              = lastWritingIndex + lastVertexNb;
        DeviceBuffer   finalVerticesPosition = DeviceBuffer::Typed<Real4>( vertexNb );
        DeviceBuffer   finalVerticesId       = DeviceBuffer::Typed<uint32_t>( vertexNb * 4 );

        impl::saveVertices<<<gridDim, blockDim>>>( context,
                                                   finalVerticesPosition.get<Real4>(),
                                                   finalVerticesId.get<uint32_t>(),
                                                   writeIndices.get<uint32_t>(),
                                                   bvh.getDeviceView() );

        VertexDiagram diagram {};
        diagram.vertices.resize( vertexNb * 4 );
        diagram.verticesId.resize( vertexNb * 4 );
        mmemcpy<MemcpyType::DeviceToHost>( diagram.vertices.data(), finalVerticesPosition.get<Real>(), vertexNb * 4 );
        mmemcpy<MemcpyType::DeviceToHost>( diagram.verticesId.data(), finalVerticesId.get<uint32_t>(), vertexNb * 4 );

        return diagram;
    }

    template<uint32_t K>
    FullDiagram AlgorithmGPU<K>::toHostFull( bool doValidation )
    {
        ContextCellOriented context {};
        context.sites     = dSites.get<Real4>();
        context.siteNb    = m_siteNb;
        context.minRadius = m_minRadius;
        context.maxRadius = m_maxRadius;

        // Storage for data structure
        context.status   = dStatus.get<uint8_t>();
        context.vertexNb = dVerticesNb.get<uint32_t>();
        context.vertices = dVertices.get<Vertex>();
        context.edges    = dEdges.get<Edge>();
        context.edgeNb   = dEdgesNb.get<uint32_t>();

        // save vertices
        DeviceBuffer writeIndices  = DeviceBuffer::Typed<uint32_t>( m_siteNb );
        auto [ gridDim, blockDim ] = KernelConfig::From( context.siteNb, impl::BlockSize );
        impl::getWriteableVertices<<<gridDim, blockDim>>>( context, writeIndices.get<uint32_t>() );
        apo::gpu::cudaCheck( "getWriteableVertices" );

        uint32_t lastVertexNb = 0;
        mmemcpy<MemcpyType::DeviceToHost>( &lastVertexNb, writeIndices.get<uint32_t>() + m_siteNb - 1, 1 );

        thrust::exclusive_scan( thrust::device,
                                writeIndices.get<uint32_t>(),
                                writeIndices.get<uint32_t>() + m_siteNb,
                                writeIndices.get<uint32_t>() );

        uint32_t lastWritingIndex = 0;
        mmemcpy<MemcpyType::DeviceToHost>( &lastWritingIndex, writeIndices.get<uint32_t>() + m_siteNb - 1, 1 );

        const uint32_t vertexNb              = lastWritingIndex + lastVertexNb;
        DeviceBuffer   finalVerticesPosition = DeviceBuffer::Typed<Real4>( vertexNb );
        DeviceBuffer   finalVerticesId       = DeviceBuffer::Typed<uint32_t>( vertexNb * 4 );

        impl::saveVertices<<<gridDim, blockDim>>>( context,
                                                   finalVerticesPosition.get<Real4>(),
                                                   finalVerticesId.get<uint32_t>(),
                                                   writeIndices.get<uint32_t>(),
                                                   bvh.getDeviceView() );

        // save closed edges
        // writeIndices               = DeviceBuffer::Typed<uint32_t>( m_siteNb ); // no need to recreate structure
        impl::getWriteableClosedEdges<<<gridDim, blockDim>>>( context, writeIndices.get<uint32_t>() );
        apo::gpu::cudaCheck( "getWriteableClosedEdges" );

        uint32_t lastEdgeNb = 0;
        mmemcpy<MemcpyType::DeviceToHost>( &lastEdgeNb, writeIndices.get<uint32_t>() + m_siteNb - 1, 1 );

        thrust::exclusive_scan( thrust::device,
                                writeIndices.get<uint32_t>(),
                                writeIndices.get<uint32_t>() + m_siteNb,
                                writeIndices.get<uint32_t>() );

        lastWritingIndex = 0;
        mmemcpy<MemcpyType::DeviceToHost>( &lastWritingIndex, writeIndices.get<uint32_t>() + m_siteNb - 1, 1 );

        const uint32_t edgeNb             = lastWritingIndex + lastEdgeNb;
        DeviceBuffer   finalEdgesPosition = DeviceBuffer::Typed<Real4>( edgeNb );
        DeviceBuffer   finalEdgesId       = DeviceBuffer::Typed<uint32_t>( edgeNb * 4 );

        impl::saveClosedEdges<<<gridDim, blockDim>>>( context,
                                                      finalEdgesPosition.get<Real4>(),
                                                      finalEdgesId.get<uint32_t>(),
                                                      writeIndices.get<uint32_t>(),
                                                      bvh.getDeviceView() );

        // output structure
        FullDiagram diagram {};
        diagram.vertices.resize( vertexNb * 4 );
        diagram.verticesId.resize( vertexNb * 4 );
        mmemcpy<MemcpyType::DeviceToHost>( diagram.vertices.data(), finalVerticesPosition.get<Real>(), vertexNb * 4 );
        mmemcpy<MemcpyType::DeviceToHost>( diagram.verticesId.data(), finalVerticesId.get<uint32_t>(), vertexNb * 4 );

        diagram.closedEdgesMin.resize( edgeNb * 4 );
        diagram.closedEdgesId.resize( edgeNb * 4 );
        mmemcpy<MemcpyType::DeviceToHost>( diagram.closedEdgesMin.data(), finalEdgesPosition.get<Real>(), edgeNb * 4 );
        mmemcpy<MemcpyType::DeviceToHost>( diagram.closedEdgesId.data(), finalEdgesId.get<uint32_t>(), edgeNb * 4 );

        return diagram;
    }

    template<uint32_t K>
    Topology AlgorithmGPU<K>::toTopology()
    {
        Topology topology {};

        std::vector<Vertex>   cellVertices = dVertices.toHost<Vertex>();
        std::vector<uint32_t> cellVertexNb = dVerticesNb.toHost<uint32_t>();
        std::vector<Edge>     cellEdges    = dEdges.toHost<Edge>();
        std::vector<uint32_t> cellEdgeNb   = dEdgesNb.toHost<uint32_t>();

        struct Bisector
        {
            uint32_t i, j;
            bool     operator<( const Bisector & b ) const { return i < b.i || ( i == b.i && j < b.j ); }
        };
        struct Trisector
        {
            uint32_t i, j, k;
            bool     operator<( const Trisector & b ) const
            {
                return i < b.i || ( i == b.i && j < b.j ) || ( i == b.i && j == b.j && k < b.k );
            }
        };
        struct Quadrisector
        {
            uint32_t i, j, k, l;
            bool     operator<( const Quadrisector & b ) const
            {
                return i < b.i || ( i == b.i && j < b.j ) || ( i == b.i && j == b.j && k < b.k )
                       || ( i == b.i && j == b.j && k == b.k && l < b.l );
            }
        };

        std::vector<uint32_t> baseIndices = bvh.indices.toHost<uint32_t>();
        for ( uint32_t i = 0; i < m_siteNb; i++ )
        {
            std::set<Bisector>     bisectors {};
            std::set<Trisector>    trisectors {};
            std::set<Quadrisector> quadrisectors {};

            const uint32_t eStart = i * impl::TotalMaxEdgeNb;
            const uint32_t eEnd   = eStart + cellEdgeNb[ i ];
            for ( uint32_t e = eStart; e < eEnd; e++ )
            {
                const Edge & edge = cellEdges[ e ];
                if ( i < edge.j && edge.j < m_siteNb )
                    bisectors.emplace( Bisector { i, edge.j } );
                if ( i < edge.k && edge.k < m_siteNb )
                    bisectors.emplace( Bisector { i, edge.k } );

                if ( i < edge.j && i < edge.k && edge.j < m_siteNb && edge.k < m_siteNb )
                    trisectors.emplace( Trisector { i, edge.j, edge.k } );
            }

            const uint32_t vStart = i * impl::TotalMaxVertexNb;
            const uint32_t vEnd   = vStart + cellVertexNb[ i ];
            for ( uint32_t v = vStart; v < vEnd; v++ )
            {
                const Vertex & vertex = cellVertices[ v ];
                if ( i < vertex.j && i < vertex.k && i < vertex.l && vertex.j < m_siteNb && vertex.k < m_siteNb
                     && vertex.l < m_siteNb )
                    quadrisectors.emplace( Quadrisector { i, vertex.j, vertex.k, vertex.l } );
            }

            topology.bisectors.reserve( topology.bisectors.size() + bisectors.size() * 2 );
            for ( const Bisector & bisector : bisectors )
            {
                topology.bisectors.emplace_back( baseIndices[ bisector.i ] );
                topology.bisectors.emplace_back( baseIndices[ bisector.j ] );
            }

            topology.trisectors.reserve( topology.trisectors.size() + trisectors.size() * 3 );
            for ( const Trisector & trisector : trisectors )
            {
                topology.trisectors.emplace_back( baseIndices[ trisector.i ] );
                topology.trisectors.emplace_back( baseIndices[ trisector.j ] );
                topology.trisectors.emplace_back( baseIndices[ trisector.k ] );
            }

            topology.quadrisectors.reserve( topology.quadrisectors.size() + quadrisectors.size() * 4 );
            for ( const Quadrisector & quadrisector : quadrisectors )
            {
                topology.quadrisectors.emplace_back( baseIndices[ quadrisector.i ] );
                topology.quadrisectors.emplace_back( baseIndices[ quadrisector.j ] );
                topology.quadrisectors.emplace_back( baseIndices[ quadrisector.k ] );
                topology.quadrisectors.emplace_back( baseIndices[ quadrisector.l ] );
            }
        }
        return topology;
    }
} // namespace apo::gpu