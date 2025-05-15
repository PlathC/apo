#include "apo/gpu/heap.hpp"
#include "apo/gpu/lbvh.cuh"

namespace apo::gpu
{
    // Based on: https://github.com/ingowald/cuBQL/blob/4c662fe2f01224b725b2afc2e3137f7263f565b6/cuBQL/queries/knn.h#L143
    template<class Predicate>
    __device__ void LBVH::View::getKnns( uint32_t            k,
                                         const Real4 * const spheres,
                                         uint32_t *          knns,
                                         float *             knnsDistance,
                                         const float4        sphere,
                                         float               startingDistance,
                                         Predicate           validate ) const
    {
        int2     stack[ 32 ];
        int2 *   stackPtr = stack;
        uint32_t nodeId   = 0; // Starting from the root
        while ( true )
        {
            while ( true )
            {
                const bool isLeaf = nodeId >= count - 1;
                if ( isLeaf )
                    break;

                const uint32_t left  = nodesLeft[ nodeId ];
                const uint32_t right = nodesRight[ nodeId ];

                const Aabbf leftBB = {
                    make_float3( nodesAabb[ left * 2 + 0 ] ),
                    make_float3( nodesAabb[ left * 2 + 1 ] ),
                };
                const Aabbf rightBB = {
                    make_float3( nodesAabb[ right * 2 + 0 ] ),
                    make_float3( nodesAabb[ right * 2 + 1 ] ),
                };

                const float dLeft     = leftBB.distance( sphere );
                const float dLeftMax  = leftBB.farthestDistance( sphere );
                const float dRight    = rightBB.distance( sphere );
                const float dRightMax = rightBB.farthestDistance( sphere );

                const bool     isLeftClosest = dLeft < dRight;
                const float    minFarthest   = isLeftClosest ? dRight : dLeft;
                const float    maxFarthest   = isLeftClosest ? dRightMax : dLeftMax;
                const uint32_t farthestChild = isLeftClosest ? right : left;
                const float    minClosest    = isLeftClosest ? dLeft : dRight;
                const float    maxClosest    = isLeftClosest ? dLeftMax : dRightMax;
                const uint32_t closestChild  = isLeftClosest ? left : right;

                const bool exploreFarthest = minFarthest < knnsDistance[ 0 ] && maxFarthest > startingDistance;
                const bool exploreClosest  = minClosest < knnsDistance[ 0 ] && maxClosest > startingDistance;
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

            const bool isLeaf = nodeId >= count - 1 && nodeId < ( 2 * count - 1 );
            if ( isLeaf )
            {
                const uint32_t j        = nodeId - ( count - 1 );
                const float4   sj       = toFloat( spheres[ j ] );
                const float    distance = sphereDistance( sphere, sj );
                if ( validate( j ) && distance < knnsDistance[ 0 ] && distance > startingDistance )
                {
                    knnsDistance[ 0 ] = distance;
                    knns[ 0 ]         = j;

                    heapify<float>( knns, knnsDistance, 0, k );
                }
            }

            bool continueTraversal = true;
            while ( true )
            {
                if ( stackPtr == stack )
                {
                    continueTraversal = false;
                    break;
                }

                --stackPtr;
                if ( __int_as_float( stackPtr->y ) > knnsDistance[ 0 ] )
                    continue;

                nodeId = stackPtr->x;
                break;
            }

            if ( !continueTraversal )
                break;
        }

        heapsort( knns, knnsDistance, k );
    }

    template<class Predicate>
    __device__ uint32_t LBVH::View::findIntersection( const Real4 * const spheres,
                                                      const float4        sphere,
                                                      Predicate           validate ) const
    {
        int2     stack[ 32 ];
        int2 *   stackPtr = stack;
        uint32_t nodeId   = 0; // Starting from the root

        uint32_t invalidating = 0xffffffff;
        while ( true )
        {
            while ( true )
            {
                const bool isLeaf = nodeId >= count - 1;
                if ( isLeaf )
                    break;

                const uint32_t left  = nodesLeft[ nodeId ];
                const uint32_t right = nodesRight[ nodeId ];

                const Aabbf leftBB = {
                    make_float3( nodesAabb[ left * 2 + 0 ] ),
                    make_float3( nodesAabb[ left * 2 + 1 ] ),
                };
                const Aabbf rightBB = {
                    make_float3( nodesAabb[ right * 2 + 0 ] ),
                    make_float3( nodesAabb[ right * 2 + 1 ] ),
                };

                const float dLeft  = leftBB.distance( sphere );
                const float dRight = rightBB.distance( sphere );

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

            const bool isLeaf = nodeId >= count - 1 && nodeId < 2 * count - 1;
            if ( isLeaf )
            {
                const uint32_t otherId = nodeId - ( count - 1 );
                if ( validate( otherId ) )
                {
                    const float4 sj       = toFloat( spheres[ otherId ] );
                    const float  distance = sphereDistance( sphere, sj );
                    if ( apo::lessThan( distance, -1e-4f ) )
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

        return invalidating;
    }
} // namespace apo::gpu
