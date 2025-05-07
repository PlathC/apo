#include <apo/core/benchmark.hpp>
#include <common/samples.hpp>
#include <common/voronota.hpp>

#include "benchmark.hpp"

int main( int, char ** )
{
    constexpr uint32_t         SampleNb      = 100;
    constexpr uint32_t         WarmupNb      = 10;
    constexpr std::string_view Configuration = "I9-13900K-RTX-4090";

    std::vector<apo::Path> dataset = {
        // ANOMALYSET
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/ANOMALYSET/ANO1_0CONNECT.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/ANOMALYSET/ANO2_0CONNECT.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/ANOMALYSET/ANO3_3CONNECT.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/ANOMALYSET/ANO4_4CONNECT.txt",

        // BALLCLOUD
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_1/BALL_1_1_10000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_1/BALL_1_1_20000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_1/BALL_1_1_30000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_1/BALL_1_1_40000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_1/BALL_1_1_50000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_1/BALL_1_1_60000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_1/BALL_1_1_70000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_1/BALL_1_1_80000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_1/BALL_1_1_90000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_1/BALL_1_1_100000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_2/BALL_1_2_10000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_2/BALL_1_2_20000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_2/BALL_1_2_30000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_2/BALL_1_2_40000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_2/BALL_1_2_50000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_2/BALL_1_2_60000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_2/BALL_1_2_70000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_2/BALL_1_2_80000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_2/BALL_1_2_90000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_2/BALL_1_2_100000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_5/BALL_1_5_10000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_5/BALL_1_5_20000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_5/BALL_1_5_30000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_5/BALL_1_5_40000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_5/BALL_1_5_50000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_5/BALL_1_5_60000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_5/BALL_1_5_70000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_5/BALL_1_5_80000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_5/BALL_1_5_90000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_5/BALL_1_5_100000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_10/BALL_1_10_10000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_10/BALL_1_10_20000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_10/BALL_1_10_30000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_10/BALL_1_10_40000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_10/BALL_1_10_50000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_10/BALL_1_10_60000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_10/BALL_1_10_70000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_10/BALL_1_10_80000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_10/BALL_1_10_90000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLCLOUD/BALL_1_10/BALL_1_10_100000.txt",

        // BALLSMALLSET
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_1000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_2000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_3000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_4000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_5000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_6000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_7000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_8000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_9000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_10000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_11000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_12000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_13000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_14000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_15000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_16000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_17000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_18000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_19000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_20000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_21000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_22000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_23000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_24000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_25000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_26000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_27000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_28000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_29000.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_30000.txt",

        // EXTREMESET
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/EXTREMESET/Ext_I_Congruent_300.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/EXTREMESET/Ext_II_Polysized_300.txt",

        // VISUALSET
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/VISUALSET/Vis_I_10.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/VISUALSET/Vis_II_5.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/VISUALSET/Vis_III_5.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/VISUALSET/Vis_IV_60.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/VISUALSET/Vis_V_20.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/VISUALSET/Vis_VI_6.txt",
        "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/VISUALSET/Vis_VII_20.txt",
    };

    std::vector<char> out {};
    fmt::format_to( std::back_inserter( out ), "Name;SiteCount;Method;Iteration;Time\n" );
    for ( const auto & path : dataset )
    {
        std::vector<apo::Real> sites  = apo::parseFromDataset( path );
        const std::size_t      siteNb = sites.size() / 4;
        const std::string      name   = std::filesystem::path( path ).stem().string();
        apo::logger::info( "Perform benchmark for {} with {} sites", name, siteNb );

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
                std::back_inserter( out ), "{};{};Voronota;{};{}\n", name, siteNb, j, voronotaSamples[ j ] );

        // Benchmark apo
        std::vector<double> apoSamples = apo::benchmark( WarmupNb, SampleNb, sites );

        // output apo
        for ( std::size_t j = 0; j < apoSamples.size(); j++ )
            fmt::format_to( std::back_inserter( out ), "{};{};Ours;{};{}\n", name, siteNb, j, apoSamples[ j ] );
    }

    auto outputCsv      = std::ofstream { fmt::format(
        "./apovsvoronota{}-dataset-warm-{}-samples-{}-{}.csv",
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
