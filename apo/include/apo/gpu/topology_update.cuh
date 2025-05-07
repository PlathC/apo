#ifndef APO_GPU_TOPOLOGY_UPDATE_CUH
#define APO_GPU_TOPOLOGY_UPDATE_CUH

#include <cub/warp/warp_merge_sort.cuh>

#include "apo/gpu/math.cuh"
#include "apo/gpu/utils.cuh"

namespace apo::gpu
{
    constexpr uint32_t WarpSize = 32;

    constexpr uint32_t ArtificialSiteNb     = 6;
    constexpr uint32_t ArtificialVerticesNb = 8;
    constexpr uint32_t ArtificialEdgesNb    = 12;

    constexpr bool     DebugCellConstruction = false;
    constexpr uint32_t DebugId               = 1008;

    // 16 bytes
    struct Edge
    {
        uint32_t j;
        uint32_t k;
        uint32_t vertexNb;
        uint32_t status : 8;      // {0: Edge is not sure to be valid, 1: Edge is known to be valid}
        uint32_t validatedNb : 8; // Count of unloaded valid number of vertex linked to this edge
    };

    // 16 bytes
    // + 16 / 32 bytes
    // = 32 / 48 bytes
    struct Vertex
    {
        // i is induced by the current cell
        uint32_t j;
        uint32_t k;
        uint32_t l;
        uint32_t type : 8;   // Solution index {0, 1}
        uint32_t status : 8; // {0: Vertex is not sure to be valid, 1: Vertex is known to be valid}
    };

    struct CellEdges
    {
        // Edge nb
        uint32_t size;
        Edge *   edges;
    };

    struct CellVertices
    {
        uint32_t size;
        Vertex * vertices;
    };

    struct NewEdge
    {
        uint32_t k : 30;
        bool     twoSolutions : 1;
        bool     closed : 1;

        __device__ bool isValid() const { return !( twoSolutions && closed ); }

        // Can't have twoSolutions and a closed trisector
        __device__ static NewEdge NoEdge() { return { 0xfffffff, true, true }; }
    };

    struct NewEdgeLess
    {
        __device__ bool operator()( const NewEdge & lhs, const NewEdge & rhs )
        {
            return lhs.k == rhs.k ? lhs.twoSolutions : lhs.k < rhs.k;
        }
    };

    enum CellStatus
    {
        InValidation   = 0x0,
        FullyValidated = 0x1,
        Buried         = 0xff
    };

    struct ContextCellOriented
    {
        uint32_t siteNb    = 0;
        Real4 *  sites     = nullptr;
        Real     minRadius = Real( 0 );
        Real     maxRadius = Real( 0 );

        uint8_t * status = nullptr;

        uint32_t * vertexNb = nullptr;
        Vertex *   vertices = nullptr;

        uint32_t * edgeNb = nullptr;
        Edge *     edges  = nullptr;
    };

    template<uint32_t MaxVerticesPerCell, uint32_t MaxEdgesPerCell>
    __global__ void initializeDataStructure( ContextCellOriented context,
                                             uint32_t * const    artificialVerticesIndices,
                                             uint32_t * const    artificialEdgesIndices );

    template<uint32_t MaxNewEdgeNb>
    using WarpMergeSortNewEdgeT = cub::WarpMergeSort<NewEdge, MaxNewEdgeNb / WarpSize, WarpSize>;

    template<uint32_t MaxVerticesPerCell, uint32_t MaxEdgesPerCell, uint32_t MaxNewEdgeNb>
    __device__ bool updateCell( cg::thread_block_tile<WarpSize>                             warp,
                                ContextCellOriented                                         context,
                                uint32_t                                                    i,
                                const Real4 &                                               si,
                                CellVertices &                                              cellVertices,
                                CellEdges &                                                 cellEdges,
                                const uint32_t                                              j,
                                const Real4 &                                               sj,
                                typename WarpMergeSortNewEdgeT<MaxNewEdgeNb>::TempStorage & tempStorage );

    template<uint32_t MaxVerticesPerCell, uint32_t MaxEdgesPerCell, uint32_t MaxNewEdgeNb, class ValidationPredicate>
    __device__ bool updateCell( cg::thread_block_tile<WarpSize>                                warp,
                                ContextCellOriented                                            context,
                                uint32_t                                                       i,
                                const Real4 &                                                  si,
                                CellVertices &                                                 cellVertices,
                                CellEdges &                                                    cellEdges,
                                const uint32_t                                                 j,
                                const Real4 &                                                  sj,
                                typename WarpMergeSortNewEdgeT<MaxEdgesPerCell>::TempStorage & tempStorage,
                                ValidationPredicate                                            predicate );

    inline __device__ Real cellRadius( cg::thread_block_tile<WarpSize> warp,
                                       ContextCellOriented             context,
                                       const Real4 &                   si,
                                       const CellVertices &            cellVertices,
                                       const CellEdges &               cellEdges );
} // namespace apo::gpu

#include "apo/gpu/topology_update.inl"

#endif // APO_GPU_TOPOLOGY_UPDATE_CUH