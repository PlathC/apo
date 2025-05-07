#ifndef APO_CAVITY_DETECTION_UTILS_HPP
#define APO_CAVITY_DETECTION_UTILS_HPP

#include <apo/gpu/memory.cuh>
#include <apo/gpu/setup.hpp>

namespace apo::gpu
{
    apo::gpu::DeviceBuffer getHitRatios( const uint32_t           vertexNb,
                                         apo::gpu::DeviceBuffer & dVertices,
                                         ConstSpan<float>         sites );

    std::vector<float> getVertices( const uint32_t           vertexNb,
                                    apo::gpu::DeviceBuffer & dVertices,
                                    apo::gpu::DeviceBuffer & hitRatios,
                                    float                    hitRatioThreshold,
                                    float                    radiusThreshold );

    uint32_t getVertices( apo::ConstSpan<apo::Real> sites, apo::gpu::DeviceBuffer & dVertices );

    struct FilterConfiguration
    {
        uint32_t vertexNb;
        float    hitRatioThreshold;
        float    radiusThreshold;
    };

    apo::gpu::DeviceBuffer getDirections( uint32_t count );

    uint32_t filterVertices( FilterConfiguration      configuration,
                             apo::gpu::DeviceBuffer & vertices,
                             apo::gpu::DeviceBuffer & hitRatios,
                             apo::gpu::DeviceBuffer & finalVertices );
} // namespace apo::gpu

#endif // APO_CAVITY_DETECTION_UTILS_HPP