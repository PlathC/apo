#ifndef APO_GPU_BENCHMARK_CUH
#define APO_GPU_BENCHMARK_CUH

#include "apo/core/benchmark.hpp"

namespace apo::gpu
{
    double timer_ms( const Benchmark::Task & task );
}

#endif // APO_GPU_BENCHMARK_CUH
