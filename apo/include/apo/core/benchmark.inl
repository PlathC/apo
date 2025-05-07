#include "apo/core/benchmark.hpp"

#ifdef _WIN32
#include <intrin.h>
#else
#if !defined( __APPLE__ ) && !defined( __CUDACC__ )
#include <x86intrin.h>
#endif // !__APPLE__
#endif

namespace apo
{
#if defined( _MSC_VER )
    template<typename T>
    void Benchmark::doNotOptimize( T const & val )
    {
        doNotOptimizeSink( &val );
    }
#else
    template<typename T>
    void Benchmark::doNotOptimize( T const & val )
    {
#if !defined( __APPLE__ ) && !defined( __CUDACC__ )
        asm volatile( "" : : "r,m"( val ) : "memory" );
#endif // __APPLE__
    }

    template<typename T>
    void Benchmark::doNotOptimize( T & val )
    {
#if defined( __clang__ ) && !defined( __APPLE ) && !defined( __CUDACC__ )
        asm volatile( "" : "+r,m"( val ) : : "memory" );
#elif !defined( __APPLE ) && !defined( __CUDACC__ )
        asm volatile( "" : "+m,r"( val ) : : "memory" );
#endif
    }
#endif

    inline Benchmark & Benchmark::iterations( const std::size_t iterations )
    {
        m_iterations = iterations;
        return *this;
    }

    inline Benchmark & Benchmark::warmups( const std::size_t warmups )
    {
        m_warmups = warmups;
        return *this;
    }

    inline Benchmark & Benchmark::timerFunction( const std::function<double( const Task & )> & timerFunction,
                                                 const std::string &                           timerUnit )
    {
        m_timerFunction = timerFunction;
        m_timerUnit     = timerUnit;
        return *this;
    }

    inline Benchmark & Benchmark::printProgress( const bool printProgress )
    {
        m_printProgress = printProgress;
        return *this;
    }

    inline Benchmark & Benchmark::printStats( const bool printStats )
    {
        m_printStats = printStats;
        return *this;
    }

    inline Benchmark & Benchmark::name( std::string name )
    {
        m_name = std::move( name );
        return *this;
    }
} // namespace apo
