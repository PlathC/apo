#ifndef APO_GPU_LBVH_CU
#define APO_GPU_LBVH_CU

#include "apo/gpu/math.cuh"
#include "apo/gpu/memory.cuh"

namespace apo::gpu
{
    struct LBVH
    {
      public:
        struct View
        {
            uint32_t count;

            const uint32_t * const indices;

            const int * const      nodesParent;
            const int * const      nodesLeft;
            const int * const      nodesRight;
            const float4 * const   nodesAabb;
            const uint32_t * const sizes;

            template<class Predicate>
            APO_DEVICE void getKnns( uint32_t            k,
                                     const Real4 * const spheres,
                                     uint32_t *          knns,
                                     float *             knnsDistance,
                                     const float4        sphere,
                                     float               startingDistance,
                                     Predicate           validate ) const;

            template<class Predicate>
            APO_DEVICE uint32_t findIntersection( const Real4 * const spheres,
                                                  const float4        sphere,
                                                  Predicate           validate ) const;
        };

        LBVH() = default;
        void       build( uint32_t count, Aabbf sceneBox, Real4 * spheres, bool sort = true );
        void       buildf( uint32_t count, Aabbf sceneBox, float4 * spheres, bool sort = true );
        LBVH::View getDeviceView() const
        {
            return {
                elementCount,
                indices.get<uint32_t>(),
                nodeParentIndices.get<int>(),
                nodeLeftIndices.get<int>(),
                nodeRightIndices.get<int>(),
                nodesAabbs.get<float4>(),
                nodesSizes.get<uint32_t>(),
            };
        }

        uint32_t     elementCount;
        DeviceBuffer indices;
        DeviceBuffer nodeParentIndices;
        DeviceBuffer nodeLeftIndices;
        DeviceBuffer nodeRightIndices;
        DeviceBuffer nodesAabbs;
        DeviceBuffer nodesSizes;
    };
} // namespace apo::gpu

#include "apo/gpu/lbvh.inl"

#endif // APO_GPU_LBVH_CU