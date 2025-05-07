#include <fstream>

#include <apo/core/utils.hpp>
#include <apo/gpu/algorithm.hpp>
#include <common/samples.hpp>
#include <common/voronota.hpp>

#include "precision.hpp"

std::vector<apo::precision::Quadrisector> getVoronotaQuadrisectors( voronota::ParallelComputationResult & result )
{
    // Merge quadruples maps
    for ( std::size_t i = 0; i < result.distributed_quadruples_maps.size(); i++ )
    {
        result.number_of_produced_quadruples += result.distributed_quadruples_maps[ i ].size();
        voronota::apollota::Triangulation::merge_quadruples_maps( result.distributed_quadruples_maps[ i ],
                                                                  result.merged_quadruples_map );
    }

    std::vector<apo::precision::Quadrisector> quadrisectors {};
    for ( const auto & value : result.merged_quadruples_map )
    {
        const voronota::apollota::Quadruple & quadruple = value.first;
        for ( auto & sphere : value.second )
        {
            quadrisectors.emplace_back( apo::precision::Quadrisector {
                static_cast<uint32_t>( quadruple.get( 0 ) ),
                static_cast<uint32_t>( quadruple.get( 1 ) ),
                static_cast<uint32_t>( quadruple.get( 2 ) ),
                static_cast<uint32_t>( quadruple.get( 3 ) ),
                sphere.x,
                sphere.y,
                sphere.z,
            } );
        }
    }

    return quadrisectors;
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

    std::vector<char> out {};
    fmt::format_to( std::back_inserter( out ), "Name;SiteCount;BadQuad;TotalQuad;BadClosedTri;TotalClosedTri\n" );

    for ( const auto & path : dataset )
    {
        std::vector<apo::Real> sites  = apo::parseFromDataset( path );
        const std::size_t      siteNb = sites.size() / 4;
        const std::string      name   = std::filesystem::path( path ).stem().string();
        apo::logger::info( "Perform precision benchmark for {} with {} sites", name, siteNb );

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

        voronota::ParallelComputationProcessingWithOpenMP::process( result, 1 );

        // Voronota
        const std::vector<apo::precision::Quadrisector> quadrisectors = getVoronotaQuadrisectors( result );
        const std::size_t badQuadNb = apo::precision::getBadQuadrisectorNb( sites, quadrisectors );

        fmt::format_to(
            std::back_inserter( out ), "{};{};{};{};{};{}\n", name, siteNb, badQuadNb, quadrisectors.size(), 0, 0 );
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
        apo::logger::info( "Perform precision benchmark for {} with {} sites", pdb, siteNb );

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

        voronota::ParallelComputationProcessingWithOpenMP::process( result, 1 );

        // Voronota
        const std::vector<apo::precision::Quadrisector> quadrisectors = getVoronotaQuadrisectors( result );
        const std::size_t badQuadNb = apo::precision::getBadQuadrisectorNb( sites, quadrisectors );

        fmt::format_to(
            std::back_inserter( out ), "{};{};{};{};{};{}\n", pdb, siteNb, badQuadNb, quadrisectors.size(), 0, 0 );
    }

    struct CloudConfiguration
    {
        uint32_t  count;
        apo::Real spreading;
        apo::Real radiiFactor;
        apo::Real radiiStart;
    };

    std::vector<CloudConfiguration> cloudTestSet = {
        CloudConfiguration { 100, 25., 2., 1. },
        CloudConfiguration { 1000, 50., 2., 1. },
        CloudConfiguration { 100, 100., 10., 0.1 },
        CloudConfiguration { 1000, 200., 10., 0.1 },
    };

    for ( const CloudConfiguration & configuration : cloudTestSet )
    {
        // Load cloud
        std::vector<apo::Real> sites = apo::getUniform(
            configuration.count, configuration.spreading, configuration.radiiFactor, configuration.radiiStart );
        apo::logger::info(
            "Perform precision benchmark for configuration of sites: {}, spreading: {}, radiiFactor: {}, radiiStart: "
            "{}",
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

        voronota::ParallelComputationProcessingWithOpenMP::process( result, 1 );

        // Voronota
        const std::vector<apo::precision::Quadrisector> quadrisectors = getVoronotaQuadrisectors( result );
        const std::size_t badQuadNb = apo::precision::getBadQuadrisectorNb( sites, quadrisectors );

        fmt::format_to( std::back_inserter( out ),
                        "{}-{}-{}-{};{};{};{};{};{}\n",
                        configuration.count,
                        configuration.spreading,
                        configuration.radiiFactor,
                        configuration.radiiStart,
                        configuration.count,
                        badQuadNb,
                        quadrisectors.size(),
                        0,
                        0 );
    }

    auto outputCsv      = std::ofstream { fmt::format( "./precision-voronota.csv" ) };
    auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
    std::copy( out.begin(), out.end(), outputIterator );

    return EXIT_SUCCESS;
}