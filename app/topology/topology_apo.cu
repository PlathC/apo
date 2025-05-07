#include <fstream>

#include <apo/core/utils.hpp>
#include <apo/gpu/algorithm.cuh>
#include <common/samples.hpp>

#include "topology.hpp"

apo::gpu::Topology getApoTopology( apo::ConstSpan<apo::Real> sites )
{
    auto cellOriented = apo::gpu::AlgorithmGPU<> { sites };
    cellOriented.build();
    return cellOriented.toTopology();
}

int main( int, char ** )
{
    std::vector<apo::Path> dataset = {
            // ANOMALYSET
            "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/ANOMALYSET/ANO1_0CONNECT.txt",
            "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/ANOMALYSET/ANO2_0CONNECT.txt",
            "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/ANOMALYSET/ANO3_3CONNECT.txt",
            "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/ANOMALYSET/ANO4_4CONNECT.txt",

            // BALLSMALLSET
            "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_1000.txt",
            "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_2000.txt",
            "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_3000.txt",
            "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_4000.txt",
            "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_5000.txt",
            "Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls/BALLSMALLSET/BALL_SMALL_6000.txt",

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

    for ( const auto & path : dataset )
    {
        std::vector<apo::Real> sites  = apo::parseFromDataset( path );
        const std::size_t      siteNb = sites.size() / 4;
        const std::string      name   = std::filesystem::path( path ).stem().string();
        apo::logger::info( "Perform topology benchmark for {} with {} sites", name, siteNb );

        // Random perturbation to ensure general position
        apo::joggle( sites, 1e-3 );

        // APO
        const apo::gpu::Topology apoTopology = getApoTopology( sites );
        const std::vector<char>  apoOut      = apo::toCsv( apoTopology );

        auto apoOutputCsv      = std::ofstream { fmt::format( "./topology-apo{}-{}.json",
#if APO_REAL_SIZE == 4
                "-fp32",
#else
                                                              "",
#endif // APO_REAL_SIZE
                                                              name ) };
        auto apoOutputIterator = std::ostream_iterator<char> { apoOutputCsv, "" };
        std::copy( apoOut.begin(), apoOut.end(), apoOutputIterator );
    }

    std::vector<apo::Path> proteinTestSet = {
            "samples/5ZCK.mmtf", // 31
            "samples/1AGA.mmtf", // 126
            "samples/3DIK.mmtf", // 219
            "samples/101M.mmtf", // 1413
            "samples/1A3F.mmtf", // 2784
    };

    for ( const apo::Path & path : proteinTestSet )
    {
        std::vector<apo::Real> sites  = apo::loadProtein( path );
        const std::size_t      siteNb = sites.size() / 4;
        const std::string      pdb    = std::filesystem::path( path ).stem().string();
        apo::logger::info( "Perform topology benchmark for {} with {} sites", pdb, siteNb );

        // Random perturbation to ensure general position
        apo::joggle( sites, 1e-3 );

        // APO
        const apo::gpu::Topology apoTopology = getApoTopology( sites );
        const std::vector<char>  apoOut      = apo::toCsv( apoTopology );

        auto apoOutputCsv      = std::ofstream { fmt::format( "./topology-apo{}-{}.json",
#if APO_REAL_SIZE == 4
                "-fp32",
#else
                                                              "",
#endif // APO_REAL_SIZE
                                                              pdb ) };
        auto apoOutputIterator = std::ostream_iterator<char> { apoOutputCsv, "" };
        std::copy( apoOut.begin(), apoOut.end(), apoOutputIterator );
    }

    struct CloudConfiguration
    {
        uint32_t  count;
        apo::Real spreading;
        apo::Real radiiFactor;
        apo::Real radiiStart;
    };

    std::vector<CloudConfiguration> cloudTestSet = {
            CloudConfiguration { 100, 25., 2., 1. },      CloudConfiguration { 1000, 50., 2., 1. },
            CloudConfiguration { 100, 100., 10., 0.1 },   CloudConfiguration { 1000, 200., 10., 0.1 },
    };

    for ( const CloudConfiguration & configuration : cloudTestSet )
    {
        // Load cloud
        std::vector<apo::Real> sites = apo::getUniform(
                configuration.count, configuration.spreading, configuration.radiiFactor, configuration.radiiStart );
        apo::logger::info(
                "Perform topology benchmark for configuration of sites: {}, spreading: {}, radiiFactor: {}, radiiStart: {}",
                configuration.count,
                configuration.spreading,
                configuration.radiiFactor,
                configuration.radiiStart );

        // Random perturbation to ensure general position
        apo::joggle( sites, 1e-3 );

        // APO
        const apo::gpu::Topology apoTopology = getApoTopology( sites );
        const std::vector<char>  apoOut      = apo::toCsv( apoTopology );

        auto apoOutputCsv = std::ofstream {
            fmt::format( "./topology-apo{}-{}-{}-{}-{}.json",
#if APO_REAL_SIZE == 4
                         "-fp32",
#else
                        "",
#endif // APO_REAL_SIZE
                         configuration.count,
                         configuration.spreading,
                         configuration.radiiFactor,
                         configuration.radiiStart )
        };
        auto apoOutputIterator = std::ostream_iterator<char> { apoOutputCsv, "" };
        std::copy( apoOut.begin(), apoOut.end(), apoOutputIterator );
    }

    return EXIT_SUCCESS;
}