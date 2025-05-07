#ifndef APO_GPU_ALGORITHM_HPP
#define APO_GPU_ALGORITHM_HPP

#include "apo/core/math.hpp"
#include "apo/core/type.hpp"

namespace apo::gpu
{
    // Host structure
    struct VertexDiagram
    {
        std::vector<Real>     vertices;   // Real4 {x, y, z, w}
        std::vector<uint32_t> verticesId; // Idu4 {i, j, k, l}
        std::vector<uint32_t> invalidationId;
    };

    // Host structure
    struct FullDiagram
    {
        std::vector<Real>     vertices;   // Real4 {x, y, z, w}
        std::vector<uint32_t> verticesId; // Idu4 {i, j, k, l}
        std::vector<uint32_t> invalidationId;

        std::vector<uint32_t> closedEdgesId;  // Idu4 {i, j, k, 0}
        std::vector<Real>     closedEdgesMin; // Real4 {x, y, z, w}
    };

    struct Topology
    {
        std::vector<uint32_t> quadrisectors; // uint4
        std::vector<uint32_t> trisectors;    // uint3
        std::vector<uint32_t> bisectors;     // uint2
    };

    class Algorithm
    {
      public:
        Algorithm( ConstSpan<Real> sites );
        virtual ~Algorithm() = default;

        virtual void          build()                     = 0;
        virtual VertexDiagram toHost( bool validate )     = 0;
        virtual FullDiagram   toHostFull( bool validate ) = 0;
        virtual Topology      toTopology()                = 0;

        ConstSpan<Real> m_sites;
        uint32_t        m_siteNb    = 0;
        Real            m_minRadius = apo::Max;
        Real            m_maxRadius = -apo::Max;
        Real            m_minX      = apo::Max;
        Real            m_minY      = apo::Max;
        Real            m_minZ      = apo::Max;
        Real            m_maxX      = -apo::Max;
        Real            m_maxY      = -apo::Max;
        Real            m_maxZ      = -apo::Max;
    };

    VertexDiagram computeApolloniusDiagram( ConstSpan<Real> sites );
    FullDiagram   computeFullApolloniusDiagram( ConstSpan<Real> sites );
} // namespace apo::gpu

#endif // APO_GPU_ALGORITHM_HPP