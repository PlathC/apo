#include "apo/core/utils.hpp"
#include "apo/gpu/algorithm.cuh"

namespace apo::gpu
{
    VertexDiagram computeApolloniusDiagram( ConstSpan<Real> sites )
    {
        apo::gpu::VertexDiagram diagram;
        auto                    cellOriented = apo::gpu::AlgorithmGPU<> { sites };
        cellOriented.build();

        diagram = cellOriented.toHost( true );

        return diagram;
    }

    FullDiagram computeFullApolloniusDiagram( ConstSpan<Real> sites )
    {
        apo::gpu::FullDiagram diagram;
        auto                  cellOriented = apo::gpu::AlgorithmGPU<> { sites };
        cellOriented.build();

        diagram = cellOriented.toHostFull( true );

        return diagram;
    }
} // namespace apo::gpu