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

    std::vector<apo::Path> proteinTestSet = {
        "samples/5ZCK.mmtf", // 31
        "samples/1AGA.mmtf", // 126
        "samples/3DIK.mmtf", // 219
        "samples/101M.mmtf", // 1413
        "samples/1A3F.mmtf", // 2784
        "samples/1A2Z.mmtf", // 7666
        "samples/8ID8.mmtf", // 8635
        "samples/7DBB.mmtf", // 17733
        "samples/7P3W.mmtf", // 37149
        "samples/7O0U.mmtf", // 55758
        "samples/1AON.mmtf", // 58870
        "samples/6QZ9.mmtf", // 71724
        "samples/3JC8.mmtf", // 107640
        "samples/4V8W.mmtf", // 123082
        "samples/7LER.mmtf", // 158430
        "samples/6RXU.mmtf", // 211834
        "samples/4V6X.mmtf", // 237685
        "samples/7CGO.mmtf", // 335722
        "samples/4V60.mmtf", // 483912
        "samples/6U42.mmtf", // 1358547
    };

    {
        std::vector<char> out {};
        fmt::format_to( std::back_inserter( out ), "Molecule;SiteCount;Method;Iteration;Time\n" );
        for ( const apo::Path & path : proteinTestSet )
        {
            std::vector<apo::Real> sites  = apo::loadProtein( path );
            const std::size_t      siteNb = sites.size() / 4;
            const std::string      pdb    = std::filesystem::path( path ).stem().string();
            apo::logger::info( "Perform benchmark for {} with {} sites", pdb, siteNb );

            // Random perturbation to ensure general position
            apo::joggle( sites, 1e-3 );

            // Convert site set to voronota format
            voronota::ParallelComputationResult result;
            result.input_spheres.resize( siteNb );
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
                fmt::format_to(
                    std::back_inserter( out ), "{};{};Voronota;{};{}\n", pdb, siteNb, j, voronotaSamples[ j ] );

            // Benchmark apo
            std::vector<double> apoSamples = apo::benchmark( WarmupNb, SampleNb, sites );

            // output apo
            for ( std::size_t j = 0; j < apoSamples.size(); j++ )
                fmt::format_to( std::back_inserter( out ), "{};{};Ours;{};{}\n", pdb, siteNb, j, apoSamples[ j ] );
        }

        auto outputCsv      = std::ofstream { fmt::format(
            "./apovsvoronota{}-proteins-warm-{}-samples-{}-{}.csv",
#if APO_REAL_SIZE == 4
                "-fp32",
#else
            "",
#endif // APO_REAL_SIZE
            WarmupNb, SampleNb, Configuration ) };
        auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
        std::copy( out.begin(), out.end(), outputIterator );
    }

    return EXIT_SUCCESS;
}
