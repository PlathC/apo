#ifndef APO_CORE_BENCHMARK_HPP
#define APO_CORE_BENCHMARK_HPP

#include <functional>
#include <string>
 
// https://github.com/google/benchmark/blob/v1.7.1/include/benchmark/benchmark.h#L221-L228
#if defined( __GNUC__ ) || defined( __clang__ )
#define BENCHMARK_ALWAYS_INLINE __attribute__( ( always_inline ) )
#elif defined( _MSC_VER ) && !defined( __clang__ )
#define BENCHMARK_ALWAYS_INLINE __forceinline
#define __func__ __FUNCTION__
#else
#define BENCHMARK_ALWAYS_INLINE
#endif

namespace apo
{
    class Benchmark
    {
      public:
        Benchmark( std::string benchmarkName = "Unnamed benchmark" );

        using Task = std::function<void()>;

// https://github.com/martinus/nanobench/blob/master/src/include/nanobench.h#L1027-L1054
#if defined( _MSC_VER )
        static void doNotOptimizeSink( void const * );
        template<typename T>
        static void doNotOptimize( T const & val );
#else
        template<typename T>
        static void doNotOptimize( T const & val );
        template<typename T>
        static void doNotOptimize( T & val );
#endif

        static double timer_ns( const Task & task ); // Nanoseconds precision timer
        static double timer_us( const Task & task ); // Microseconds precision timer
        static double timer_ms( const Task & task ); // Milliseconds precision timer

        inline Benchmark & iterations( const std::size_t iterations ); // Sets the number of iterations
        inline Benchmark & warmups( const std::size_t warmups );       // Sets the number of warmups iterations
        inline Benchmark & timerFunction(
            const std::function<double( const Task & )> & timerFunction,
            const std::string &                           timerUnit
            = "" ); // Sets the timer function and unit (optional, used if stats are printed)
        inline Benchmark & printProgress( const bool printProgress
                                          = true );                    // Sets if progress is printed during benchmark
        inline Benchmark & printStats( const bool printStats = true ); // Sets if stats are computed and printed
        inline Benchmark & name( std::string name );                   // Sets the displayed name of the benchmark

        std::vector<double> run( const Task & task ) const; // Runs the benchmark

      private:
        std::string                           m_timerUnit = "ms";
        std::string                           m_name      = "";
        std::size_t                           m_iterations { 10 };
        std::size_t                           m_warmups { 0 };
        std::function<double( const Task & )> m_timerFunction {};
        bool                                  m_printProgress { true };
        bool                                  m_printStats { false };

        std::vector<double> runInternal( const Task & task ) const;      // Runs the benchmark
        std::vector<double> runPrintInternal( const Task & task ) const; // Runs the benchmark and prints data
    };
} // namespace apo

#include "apo/core/benchmark.inl"

#endif // APO_CORE_BENCHMARK_HPP
