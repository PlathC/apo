#include <apo/core/benchmark.hpp>
#include <apo/gpu/algorithm.cuh>
#include <apo/gpu/benchmark.cuh>

#include "benchmark.hpp"

namespace apo
{
    std::vector<double> benchmark( const uint32_t warmupNb, const uint32_t sampleNb, apo::ConstSpan<apo::Real> sites )
    {
        return apo::Benchmark( "apo" )
            .timerFunction( apo::gpu::timer_ms )
            .warmups( warmupNb )
            .iterations( sampleNb )
            .printStats()
            .run(
                [ & ]
                {
                    apo::gpu::AlgorithmGPU algorithm { sites };
                    algorithm.build();
                } );
    }
} // namespace apo