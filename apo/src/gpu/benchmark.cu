#include "apo/gpu/benchmark.cuh"

namespace apo::gpu
{
    double timer_ms( const Benchmark::Task & task )
    {
        cudaEvent_t start, stop;
        cudaEventCreate( &start );
        cudaEventCreate( &stop );

        cudaEventRecord( start );

        task();

        cudaEventRecord( stop );

        cudaEventSynchronize( stop );
        float milliseconds = 0;
        cudaEventElapsedTime( &milliseconds, start, stop );

        cudaEventDestroy( start );
        cudaEventDestroy( stop );
        return milliseconds;
    }
} // namespace apo::gpu
