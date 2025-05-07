#include "apo/gpu/topology_update.cuh"

namespace apo::gpu
{
    template<uint32_t MaxVerticesPerCell, uint32_t MaxEdgesPerCell>
    __global__ void initializeDataStructure( ContextCellOriented context,
                                             uint32_t * const    artificialVerticesIndices,
                                             uint32_t * const    artificialEdgesIndices )
    {
        const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= context.siteNb )
            return;

        CellEdges cellEdges = {
            context.edgeNb[ id ],
            context.edges + id * MaxEdgesPerCell,
        };

        CellVertices cellVertices = {
            context.vertexNb[ id ],
            context.vertices + id * MaxVerticesPerCell,
        };

        for ( uint32_t avi = 0; avi < ArtificialVerticesNb; avi++ )
        {
            const uint32_t j = artificialVerticesIndices[ avi * 4 + 0 ];
            const uint32_t k = artificialVerticesIndices[ avi * 4 + 1 ];
            const uint32_t l = artificialVerticesIndices[ avi * 4 + 2 ];
            const uint32_t t = artificialVerticesIndices[ avi * 4 + 3 ];

            cellVertices.vertices[ avi ] = { j, k, l, t, 0 }; // Already ordered
        }

        context.vertexNb[ id ] = ArtificialVerticesNb;

        // No needs for synchronization, writing location is not the same
        for ( uint32_t aei = 0; aei < ArtificialEdgesNb; aei++ )
        {
            const uint32_t j = artificialEdgesIndices[ aei * 2 + 0 ];
            const uint32_t k = artificialEdgesIndices[ aei * 2 + 1 ];

            cellEdges.edges[ aei ] = { j, k, 2, 0 };
        }

        context.edgeNb[ id ] = ArtificialEdgesNb;
        context.status[ id ] = 0;
    }

    template<uint32_t MaxVerticesPerCell, uint32_t MaxEdgesPerCell, uint32_t MaxNewEdgeNb, class ValidationPredicate>
    __device__ bool updateCell( cg::thread_block_tile<WarpSize>                             warp,
                                ContextCellOriented                                         context,
                                uint32_t                                                    i,
                                const Real4 &                                               si,
                                CellVertices &                                              cellVertices,
                                CellEdges &                                                 cellEdges,
                                const uint32_t                                              j,
                                const Real4 &                                               sj,
                                typename WarpMergeSortNewEdgeT<MaxNewEdgeNb>::TempStorage & tempStorage,
                                ValidationPredicate                                         predicate )
    {
        static_assert( MaxNewEdgeNb % WarpSize == 0, "MaxNewEdgeNb must be a multiple of WarpSize" );
        constexpr uint32_t MaxNewEdgesPerThread = MaxNewEdgeNb / WarpSize;

        if ( DebugCellConstruction && i == DebugId && threadIdx.x % warpSize == 0 )
            printf( "Neighbor --> %d ([%.6f, %.6f, %.6f, %.6f])\n", j, sj.x, sj.y, sj.z, sj.w );

        if ( cellVertices.size == 0 && cellEdges.size == 0 )
            return false;

        const Real ri      = si.w;
        const Real rj      = sj.w;
        const Real bigger  = apo::max( ri, rj );
        const Real smaller = apo::min( ri, rj );

        // Check for inclusion
        if ( apo::lessThan( length( Real3 { si.x, si.y, si.z } - Real3 { sj.x, sj.y, sj.z } ), bigger - smaller ) )
        {
            if ( apo::lessThan( rj, ri ) )
                return false;

            cellVertices.size = 0;
            cellEdges.size    = 0;
            return false;
        }

        NewEdge newEdges[ MaxNewEdgesPerThread ];
        for ( uint32_t ne = 0; ne < MaxNewEdgesPerThread; ne++ )
            newEdges[ ne ] = NewEdge::NoEdge();

        // Check for intersection between vertices and aj
        const uint32_t baseVertexNb     = cellVertices.size;
        const uint32_t baseVertexStepNb = ( baseVertexNb / warpSize ) + 1;
        const uint32_t baseEdgeNb       = cellEdges.size;
        const uint32_t baseEdgeStepNb   = ( baseEdgeNb / warpSize ) + 1;

        for ( uint32_t vs = 0; vs < baseVertexStepNb; vs++ )
        {
            const uint32_t id     = warp.thread_rank() + warpSize * vs;
            Vertex &       vertex = cellVertices.vertices[ id ];

            Real4 position = Real4 { 0., 0., 0., 0. };
            if ( id < cellVertices.size )
            {
                const Real4 sk = context.sites[ vertex.j ];
                const Real4 sl = context.sites[ vertex.k ];
                const Real4 sm = context.sites[ vertex.l ];

                Real4 x[ 2 ];
                apo::gpu::quadrisector( si, sk, sl, sm, x[ 0 ], x[ 1 ] );

                position = x[ vertex.type ];
            }

            const bool isInvalid = id < baseVertexNb && intersect( position, sj );
            if ( DebugCellConstruction && i == DebugId && isInvalid )
                printf( "Removing {%d, %d, %d} at {%.6f, %.6f, %.6f, %.6f}\n",
                        vertex.j,
                        vertex.k,
                        vertex.l,
                        position.x,
                        position.y,
                        position.z,
                        position.w );

            if ( isInvalid )
                vertex.status = 0xff;

            const uint32_t invalidVertices = warp.ballot( isInvalid );
            apo::gpu::forEach( invalidVertices,
                               [ & ]( uint8_t v )
                               {
                                   const Vertex vertex = cellVertices.vertices[ warpSize * vs + v ];
                                   for ( uint8_t es = 0; es < baseEdgeStepNb; es++ )
                                   {
                                       const uint8_t id = warp.thread_rank() + warp.size() * es;
                                       if ( id < cellEdges.size )
                                       {
                                           Edge & edge = cellEdges.edges[ id ];

                                           // Easy comparison since indices are ordered
                                           const bool e1 = edge.j == vertex.j && edge.k == vertex.k;
                                           const bool e2 = edge.j == vertex.j && edge.k == vertex.l;
                                           const bool e3 = edge.j == vertex.k && edge.k == vertex.l;
                                           if ( e1 || e2 || e3 )
                                           {
                                               edge.vertexNb--;

                                               if ( DebugCellConstruction && i == DebugId && edge.vertexNb == 0 )
                                                   printf( "Removing {%d, %d}\n", edge.j, edge.k );
                                           }
                                       }
                                   }
                               } );
        }

        static_assert( MaxEdgesPerCell / WarpSize <= sizeof( uint8_t ) * 8,
                       "MaxEdgesPerCell is too large compared to elliptic edge bitset of size 8." );

        uint8_t  closedEllipticEdge = 0;
        uint32_t newEdgeNb          = 0;
        for ( uint32_t es = 0; es < baseEdgeStepNb; es++ )
        {
            const uint32_t e       = warp.thread_rank() + warpSize * es;
            const Edge     edge    = cellEdges.edges[ e ];
            bool           hasEdge = e < baseEdgeNb;

            const uint32_t k = edge.j;
            const uint32_t l = edge.k;

            uint8_t mask = 0;
            Real4   sk;
            Real4   sl;

            Real4 quadrisectors[ 2 ];
            if ( hasEdge )
            {
                sk = context.sites[ k ];
                sl = context.sites[ l ];

                mask = apo::gpu::quadrisector( si, sj, sk, sl, quadrisectors[ 0 ], quadrisectors[ 1 ] );
                hasEdge &= mask != 0;
            }

            // If no vertices are possible between the closed edge and the new neighbor, the closed edge is fully
            // invalidated by the bisector iff one of its point is invalidated.
            if ( e < baseEdgeNb && mask == 0 && edge.vertexNb == 0xffffffff )
            {
                apo::gpu::trisectorVertex( si, sk, sl, quadrisectors[ 0 ], quadrisectors[ 0 ] );
                if ( apo::gpu::intersect( quadrisectors[ 0 ], sj ) )
                    cellEdges.edges[ e ].vertexNb = 0;
            }

            // validation
            uint8_t valid = mask; // Two booleans
            apo::gpu::forEach( warp.ballot( hasEdge ),
                               [ & ]( uint8_t t )
                               {
                                   uint8_t        currentMask = warp.shfl( mask, t );
                                   const uint32_t currentK    = warp.shfl( k, t );
                                   const uint32_t currentL    = warp.shfl( l, t );
                                   while ( currentMask != 0 )
                                   {
                                       const uint8_t nv = __ffs( currentMask ) - 1;
                                       currentMask &= ( ~( 1u << ( nv ) ) );

                                       const Real4 currentQuadrisector = warp.shfl( quadrisectors[ nv ], t );
                                       bool        currentValid = predicate( currentQuadrisector, currentK, currentL );
                                       const bool  isValid      = warp.all( currentValid );
                                       if ( warp.thread_rank() == t )
                                           valid &= ~( !isValid << nv );
                                   }
                               } );

            const uint32_t newVertexNb = __popc( valid );
            hasEdge &= newVertexNb > 0;
            if ( hasEdge )
            {
                Edge & currentEdge = cellEdges.edges[ e ];
                if ( currentEdge.vertexNb == 0xffffffff )
                {
                    if ( DebugCellConstruction && DebugId == i )
                        printf( "Resetting {%d, %d}\n", currentEdge.j, currentEdge.k );
                    currentEdge.vertexNb = 0;
                }

                currentEdge.vertexNb += newVertexNb;
            }

            if ( DebugCellConstruction && i == DebugId && hasEdge )
                printf(
                    "edge = %d, valid = %d, hasEdge = %d, %d valid new vertices at {%d, %d} -> [%.6f, %.6f, %.6f, "
                    "%.6f] / [%.6f, %.6f, %.6f, %.6f]\n",
                    e,
                    valid,
                    hasEdge,
                    __popc( valid ),
                    edge.j,
                    edge.k,
                    quadrisectors[ 0 ].x,
                    quadrisectors[ 0 ].y,
                    quadrisectors[ 0 ].z,
                    quadrisectors[ 0 ].w,
                    quadrisectors[ 1 ].x,
                    quadrisectors[ 1 ].y,
                    quadrisectors[ 1 ].z,
                    quadrisectors[ 1 ].w );

            warp.sync();
            uint32_t       writingOffset           = warpInclusiveScan<uint32_t, WarpSize>( warp, newVertexNb );
            const uint32_t currentTotalNewVertexNb = warp.shfl( writingOffset, warp.size() - 1 );
            writingOffset -= newVertexNb;

            if ( DebugCellConstruction && i == DebugId && threadIdx.x % warpSize == 0 )
                printf( "currentTotalNewVertexNb = %d\n", currentTotalNewVertexNb );

            uint32_t jj = j, kk = k, ll = l;
            if ( jj > ll )
                apo::gpu::sswap( jj, ll );
            if ( jj > kk )
                apo::gpu::sswap( jj, kk );
            if ( kk > ll )
                apo::gpu::sswap( kk, ll );

            uint32_t vertexValid = valid;
            while ( vertexValid != 0 )
            {
                const uint8_t nv = static_cast<uint8_t>( __ffs( vertexValid ) - 1 );
                vertexValid &= ~( 1u << nv );

                cellVertices.vertices[ cellVertices.size + writingOffset ] = Vertex { jj, kk, ll, nv, 0 };
                if ( DebugCellConstruction && i == DebugId )
                    printf( "Writing {%d, %d, %d, %d} at %d\n", jj, kk, ll, nv, cellVertices.size + writingOffset );

                writingOffset++;
            }

            apo::gpu::forEach( warp.ballot( valid ),
                               [ & ]( uint8_t tid )
                               {
                                   uint32_t       currentValid       = warp.shfl( valid, tid );
                                   const uint32_t currentNewVertexNb = warp.shfl( newVertexNb, tid );
                                   const uint32_t currentK           = warp.shfl( k, tid );
                                   const uint32_t currentL           = warp.shfl( l, tid );
                                   while ( currentValid != 0 )
                                   {
                                       const uint8_t nv = static_cast<uint8_t>( __ffs( currentValid ) - 1 );
                                       currentValid &= ~( 1u << nv );

                                       // Add new edges, there is exactly two new edges starting from a vertex
                                       if ( newEdgeNb >= MaxNewEdgeNb )
                                           printf( "Too much new edges, increase maximum new edge number." );

                                       if ( newEdgeNb / MaxNewEdgesPerThread == warp.thread_rank() )
                                       {
                                           newEdges[ newEdgeNb % MaxNewEdgesPerThread ] = NewEdge {
                                               currentK,
                                               currentNewVertexNb == 2,
                                               false,
                                           };
                                       }
                                       newEdgeNb++;

                                       if ( newEdgeNb >= MaxNewEdgeNb )
                                           printf( "Too much new edges, increase maximum new edge number.\n" );

                                       if ( newEdgeNb / MaxNewEdgesPerThread == warp.thread_rank() )
                                       {
                                           newEdges[ newEdgeNb % MaxNewEdgesPerThread ] = NewEdge {
                                               currentL,
                                               currentNewVertexNb == 2,
                                               false,
                                           };
                                       }
                                       newEdgeNb++;
                                   }
                               } );

            cellVertices.size += currentTotalNewVertexNb;

            // A closed edge can only be produced by a new neighbor creating no vertices (no solutions)
            // and that has invalidated vertices or create an elliptic edge with an artificial site
            closedEllipticEdge |= ( e < baseEdgeNb && mask == 0 ) << es;
        }

        // Handle closed elliptic edges search
        if ( warp.any( closedEllipticEdge ) )
        {
            for ( uint32_t es = 0; es < baseEdgeStepNb; es++ )
            {
                bool canHaveClosedElliptic = ( closedEllipticEdge >> es ) & 1;
                if ( !warp.any( canHaveClosedElliptic ) )
                    continue;

                const uint32_t e       = warp.thread_rank() + warpSize * es;
                const Edge     edge    = cellEdges.edges[ e ];
                bool           hasEdge = e < baseEdgeNb;
                const uint32_t k       = edge.j;
                const uint32_t l       = edge.k;

                Real4 sk;
                Real4 sl;
                if ( hasEdge )
                {
                    sk = context.sites[ k ];
                    sl = context.sites[ l ];
                }

                const uint32_t siteIds[ 2 ] = { k, l };
                const Real4    site[ 2 ]    = { sk, sl };

                for ( uint8_t s = 0; s < 2; s++ )
                {
                    const uint32_t id                           = siteIds[ s ];
                    bool           canCurrentHaveClosedElliptic = canHaveClosedElliptic;
                    if ( DebugCellConstruction && i == DebugId && canCurrentHaveClosedElliptic )
                        printf( "a - (%d, %d) Testing {%d} -> canCurrentHaveClosedElliptic = %d\n",
                                k,
                                l,
                                id,
                                canCurrentHaveClosedElliptic );

                    // Assert that candidates to create an elliptic trisector has no vertices on other edges
                    apo::gpu::forEach( //
                        warp.ballot( canCurrentHaveClosedElliptic ),
                        [ & ]( uint8_t t )
                        {
                            const uint32_t currentId = warp.shfl( id, t );

                            bool shareVertex = false;
                            for ( uint8_t ne = 0; ne < MaxNewEdgesPerThread && !shareVertex; ne++ )
                            {
                                const uint32_t other = newEdges[ ne ].k;
                                shareVertex          = warp.any( currentId == other );
                            }

                            if ( warp.thread_rank() == t )
                                canCurrentHaveClosedElliptic &= !shareVertex;
                        } );

                    if ( DebugCellConstruction && i == DebugId && canCurrentHaveClosedElliptic )
                        printf( "b - Testing {%d} -> canCurrentHaveClosedElliptic = %d\n",
                                id,
                                canCurrentHaveClosedElliptic );

                    Real4   vertices[ 2 ];
                    uint8_t count = 0;
                    if ( canCurrentHaveClosedElliptic )
                        count = trisectorVertex( si, sj, site[ s ], vertices[ 0 ], vertices[ 1 ] );

                    // If no vertex are possible but
                    canCurrentHaveClosedElliptic &= ( count == 2 );

                    if ( DebugCellConstruction && i == DebugId && canCurrentHaveClosedElliptic )
                        printf( "c - Testing {%d} -> canCurrentHaveClosedElliptic = %d with %d\n",
                                id,
                                canCurrentHaveClosedElliptic,
                                siteIds[ s ] );

                    apo::gpu::forEach(
                        warp.ballot( canCurrentHaveClosedElliptic ),
                        [ & ]( uint8_t t )
                        {
                            const uint32_t currentId     = warp.shfl( id, t );
                            const Real4    currentVertex = warp.shfl( vertices[ 0 ], t );
                            const bool     currentValid  = predicate( currentVertex, currentId, i );
                            const bool     isValid       = warp.all( currentValid );

                            if ( !isValid )
                                return;

                            if ( DebugCellConstruction && i == DebugId && t == warp.thread_rank() )
                                printf(
                                    "%d - %d - es = %d- d - Testing {%d} with j = %d -> canCurrentHaveClosedElliptic = "
                                    "%d\n",
                                    i,
                                    warp.thread_rank(),
                                    es,
                                    id,
                                    j,
                                    canCurrentHaveClosedElliptic );

                            if ( newEdgeNb >= MaxNewEdgeNb )
                                printf( "Too much new edges, increase maximum new edge number." );

                            const uint32_t neighbor = warp.shfl( siteIds[ s ], t );
                            if ( ( newEdgeNb / MaxNewEdgesPerThread ) == warp.thread_rank() )
                                newEdges[ newEdgeNb % MaxNewEdgesPerThread ] = NewEdge { neighbor, false, true };

                            newEdgeNb++;
                        } );
                }
            }
        }

        // Find unique values of new edges
        warp.sync();
        WarpMergeSortNewEdgeT<MaxNewEdgeNb>( tempStorage ).Sort( newEdges, NewEdgeLess() );
        static_assert( MaxNewEdgesPerThread <= sizeof( uint32_t ) * 8,
                       "MaxNewEdgesPerThread is too large compared to its bitset of size 32." );

        // Find first occurrence of every new edges
        NewEdge next = warp.shfl_down( newEdges[ 0 ], 1 );

        const uint32_t lastId = MaxNewEdgesPerThread - 1;
        uint32_t       isLast = ( ( warp.thread_rank() == ( warp.size() - 1 ) || next.k != newEdges[ lastId ].k )
                            && newEdges[ lastId ].isValid() )
                          << lastId;
        for ( uint8_t ne = 0; ne < MaxNewEdgesPerThread - 1; ne++ )
        {
            next                  = newEdges[ ne + 1 ];
            const NewEdge current = newEdges[ ne ];

            isLast |= ( ( current.k != next.k ) && current.isValid() ) << ne;
        }

        // Write new edges
        warp.sync();
        uint32_t       writingOffset         = warpInclusiveScan<uint32_t, WarpSize>( warp, __popc( isLast ) );
        const uint32_t currentTotalNewEdgeNb = warp.shfl( writingOffset, warp.size() - 1 );

        writingOffset -= __popc( isLast );

        // Segmented scan to count the number of vertex per new edges
        // Reference: https://www.mgarland.org/files/papers/nvr-2008-003.pdf
        const uint32_t count = apo::gpu::clz( isLast ) - ( sizeof( isLast ) * 8 - MaxNewEdgesPerThread );

        // If the first flag is false, we can accumulate over it and need previous data
        uint32_t minIndex = 0;
        for ( uint8_t id = 1; id < warpSize; id *= 2 )
        {
            // If the current flag set contain a 1, we do not accumulate over the current thread -> reset counter
            const uint32_t otherValue = warp.shfl_up( isLast ? warp.thread_rank() : minIndex, id );
            if ( warp.thread_rank() >= id )
                minIndex = max( otherValue, minIndex );
        }

        // If all flags a zero, current threads send 2
        // If a flag is nnz, it sends the count of trailing zero
        uint32_t value = count;
        for ( uint8_t id = 1; id < warpSize; id *= 2 )
        {
            // If the flag contains nnz, we do not accumulate over the current thread -> reset count
            const uint32_t otherValue = warp.shfl_up( isLast ? count : value, id );
            if ( warp.thread_rank() >= minIndex + id )
                value = otherValue + value;
        }

        uint32_t before = value - count;
        for ( uint8_t ne = 0; ne < MaxNewEdgesPerThread; ne++ )
        {
            const uint32_t other = newEdges[ ne ].k;

            const bool isClosed = newEdges[ ne ].closed;
            const bool toAdd    = isLast & ( 1 << ne );
            if ( !toAdd )
            {
                before++;
                continue;
            }

            if ( DebugCellConstruction && i == DebugId )
                printf( "Inserting {%d, %d}, vertexNb = %d\n", other, j, isClosed ? 0xffffffff : before + 1 );

            cellEdges.edges[ cellEdges.size + writingOffset ] = {
                apo::min( j, other ),
                apo::max( j, other ),
                isClosed ? 0xffffffff : before + 1,
                0,
            };

            before = 0;
            writingOffset++;
        }

        cellEdges.size += currentTotalNewEdgeNb;

        if ( DebugCellConstruction && threadIdx.x % warpSize == 0 && i == DebugId )
            printf( "new edges nb = %d\n", currentTotalNewEdgeNb );

        warp.sync();
        if ( cellVertices.size > 0 )
        {
            const uint32_t removedVertexNb = warpCompact<WarpSize>( //
                warp,
                cellVertices.vertices,
                cellVertices.size,
                [ & ]( uint32_t id, const Vertex & vertex ) { return vertex.status == 0xff && id < baseVertexNb; } );
            cellVertices.size -= removedVertexNb;

            if ( DebugCellConstruction && i == DebugId && threadIdx.x % warpSize == 0 )
                printf( "%d Removed %d vertices\n", blockIdx.x * blockDim.x + threadIdx.x, removedVertexNb );
        }

        // Edge compaction
        warp.sync();

        const uint32_t removedEdgeNb = warpCompact<WarpSize>( //
            warp,
            cellEdges.edges,
            cellEdges.size,
            [ & ]( uint32_t id, const Edge & edge ) { return edge.vertexNb == 0 && id < baseEdgeNb; } );
        cellEdges.size -= removedEdgeNb;

        if ( DebugCellConstruction && i == DebugId && threadIdx.x % warpSize == 0 )
            printf( "Removed %d edges\n", removedEdgeNb );

        if ( DebugCellConstruction && i == DebugId && threadIdx.x % warpSize == 0 )
            printf( "---- State: #vertex = %d, #edge = %d\n", cellVertices.size, cellEdges.size );
        if ( cellEdges.size >= MaxEdgesPerCell && threadIdx.x % warpSize == 0 )
            printf( "Error on cell %d cellEdges.size = %d\n", i, cellEdges.size );
        if ( cellVertices.size >= MaxVerticesPerCell && threadIdx.x % warpSize == 0 )
            printf( "Error on cell %d cellVertices.size = %d\n", i, cellVertices.size );

        return currentTotalNewEdgeNb != 0;
    }

    template<uint32_t MaxVerticesPerCell, uint32_t MaxEdgesPerCell, uint32_t MaxNewEdgeNb>
    inline __device__ bool updateCell( cg::thread_block_tile<WarpSize>                             warp,
                                       ContextCellOriented                                         context,
                                       uint32_t                                                    i,
                                       const Real4 &                                               si,
                                       CellVertices &                                              cellVertices,
                                       CellEdges &                                                 cellEdges,
                                       const uint32_t                                              j,
                                       const Real4 &                                               sj,
                                       typename WarpMergeSortNewEdgeT<MaxNewEdgeNb>::TempStorage & tempStorage )
    {
        return updateCell<MaxVerticesPerCell, MaxEdgesPerCell, MaxNewEdgeNb>(
            warp,
            context,
            i,
            si,
            cellVertices,
            cellEdges,
            j,
            sj,
            tempStorage,
            [ & ]( const Real4 & sphere, const uint32_t skip1, const uint32_t skip2 )
            {
                bool valid = true;
                for ( uint8_t e = warp.thread_rank(); e < cellEdges.size; e += WarpSize )
                {
                    const Edge & edge = cellEdges.edges[ e ];
                    if ( edge.j != skip1 && edge.j != skip2 )
                        valid &= !intersect( context.sites[ edge.j ], sphere );

                    if ( edge.k != skip1 && edge.k != skip2 )
                        valid &= !intersect( context.sites[ edge.k ], sphere );
                }

                return valid;
            } );
    }

    inline __device__ Real cellRadius( cg::thread_block_tile<WarpSize> warp,
                                       ContextCellOriented             context,
                                       const Real4 &                   si,
                                       CellVertices &                  cellVertices,
                                       const CellEdges &               cellEdges )
    {
        Real maxRadius = 0;

        warp.sync();
        const uint32_t baseVertexStepNb = ( cellVertices.size / WarpSize ) + 1;
        for ( uint32_t stride = 0; stride < baseVertexStepNb; stride++ )
        {
            const uint32_t v = warpSize * stride + warp.thread_rank();

            Real currentRadius = Real( 0. );
            if ( v < cellVertices.size )
            {
                const Vertex vertex = cellVertices.vertices[ v ];
                const Real4  sj     = context.sites[ vertex.j ];
                const Real4  sk     = context.sites[ vertex.k ];
                const Real4  sl     = context.sites[ vertex.l ];

                Real4 x[ 2 ];
                apo::gpu::quadrisector( si, sj, sk, sl, x[ 0 ], x[ 1 ] );

                currentRadius = x[ vertex.type ].w;
            }
            warp.sync();

            const Real currentMax = apo::gpu::warpMax<Real, WarpSize>( warp, currentRadius );
            maxRadius             = apo::max( maxRadius, currentMax );
        }

        const uint32_t baseEdgeStepNb = ( cellEdges.size / WarpSize ) + 1;
        for ( uint32_t stride = 0; stride < baseEdgeStepNb; stride++ )
        {
            const uint32_t e = warpSize * stride + warp.thread_rank();

            Real     currentMaxRadius = Real( 0. );
            uint32_t vertexCount      = 0;
            if ( e < cellEdges.size )
            {
                const Edge  edge = cellEdges.edges[ e ];
                const Real4 sj   = context.sites[ edge.j ];
                const Real4 sk   = context.sites[ edge.k ];

                Real4 x[ 2 ];
                vertexCount = apo::gpu::trisectorVertex( si, sj, sk, x[ 0 ], x[ 1 ] );
                if ( vertexCount == 2 )
                    currentMaxRadius = apo::max( x[ 0 ].w, x[ 1 ].w );
            }
            warp.sync();

            const Real currentMax = apo::gpu::warpMax<Real, WarpSize>( warp, currentMaxRadius );
            maxRadius             = apo::max( maxRadius, currentMax );
        }

        return maxRadius;
    }
} // namespace apo::gpu