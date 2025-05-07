#include <fstream>
#include <iterator>

#include <apo/core/benchmark.hpp>
#include <apo/core/logger.hpp>
#include <common/samples.hpp>
#include <common/voronota.hpp>

#include "benchmark.hpp"

int main( int, char ** )
{
    constexpr uint32_t         SampleNb      = 100;
    constexpr uint32_t         WarmupNb      = 10;
    constexpr std::string_view Configuration = "I9-13900K-RTX-4090";

    struct CloudConfiguration
    {
        uint32_t  count;
        apo::Real spreading;
        apo::Real radiiFactor;
        apo::Real radiiStart;
    };

    std::vector<CloudConfiguration> cloudTestSet = {
        CloudConfiguration { 100, 25., 2., 1. },      CloudConfiguration { 1000, 50., 2., 1. },
        CloudConfiguration { 10000, 250., 2., 1. },   CloudConfiguration { 100000, 500., 2., 1. },

        CloudConfiguration { 100, 100., 10., 0.1 },   CloudConfiguration { 1000, 200., 10., 0.1 },
        CloudConfiguration { 10000, 500., 10., 0.1 }, CloudConfiguration { 100000, 1000., 10., 0.1 },
    };

    std::vector<char> out {};
    fmt::format_to( std::back_inserter( out ), "SiteCount;Spreading;radiiFactor;radiiStart;Method;Iteration;Time\n" );
    for ( const CloudConfiguration & configuration : cloudTestSet )
    {
        // Load cloud
        std::vector<apo::Real> sites = apo::getUniform(
            configuration.count, configuration.spreading, configuration.radiiFactor, configuration.radiiStart );
        apo::logger::info(
            "Perform benchmark for configuration of sites: {}, spreading: {}, radiiFactor: {}, radiiStart: {}",
            configuration.count,
            configuration.spreading,
            configuration.radiiFactor,
            configuration.radiiStart );

        // Random perturbation to ensure general position
        apo::joggle( sites, 1e-3 );

        // Convert site set to voronota format
        voronota::ParallelComputationResult result;
        result.input_spheres.resize( configuration.count );
        for ( std::size_t s = 0; s < result.input_spheres.size(); s++ )
        {
            result.input_spheres[ s ].x = sites[ s * 4 + 0 ];
            result.input_spheres[ s ].y = sites[ s * 4 + 1 ];
            result.input_spheres[ s ].z = sites[ s * 4 + 2 ];
            result.input_spheres[ s ].r = sites[ s * 4 + 3 ];
        }

        // Benchmark voronota
        const std::vector<double> voronotaSamples
            = apo::Benchmark( "Voronota" )
                  .timerFunction( apo::Benchmark::timer_ms )
                  .warmups( WarmupNb )
                  .iterations( SampleNb )
                  .printStats()
                  .run( [ & ] { voronota::ParallelComputationProcessingWithOpenMP::process( result ); } );

        // output voronota
        for ( std::size_t j = 0; j < voronotaSamples.size(); j++ )
            fmt::format_to( std::back_inserter( out ),
                            "{};{};{};{};Voronota;{};{}\n",
                            configuration.count,
                            configuration.spreading,
                            configuration.radiiFactor,
                            configuration.radiiStart,
                            j,
                            voronotaSamples[ j ] );

        // Benchmark apo
        std::vector<double> apoSamples = apo::benchmark( WarmupNb, SampleNb, sites );

        // output apo
        for ( std::size_t j = 0; j < apoSamples.size(); j++ )
            fmt::format_to( std::back_inserter( out ),
                            "{};{};{};{};Ours;{};{}\n",
                            configuration.count,
                            configuration.spreading,
                            configuration.radiiFactor,
                            configuration.radiiStart,
                            j,
                            apoSamples[ j ] );
    }

    auto outputCsv      = std::ofstream { fmt::format(
        "./apovsvoronota{}-clouds-warm-{}-samples-{}-{}.csv",
#if APO_REAL_SIZE == 4
            "-fp32",
#else
        "",
#endif // APO_REAL_SIZE
        WarmupNb, SampleNb, Configuration ) };
    auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
    std::copy( out.begin(), out.end(), outputIterator );

    return EXIT_SUCCESS;
}