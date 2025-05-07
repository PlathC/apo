#include <Eigen/Core>
#include <fstream>
#include <iostream>

#include <apo/core/type.hpp>
#include <apo/core/utils.hpp>
#include <apo/gpu/algorithm.hpp>
#include <common/samples.hpp>
#include <implicit_functions.h>
#include <material_interface.h>
#include <simplicial_arrangement/lookup_table.h>

#include "topology.hpp"

using namespace simplicial_arrangement;

apo::gpu::Topology getRobustTopology( apo::ConstSpan<apo::Real> sites, std::string meshPath )
{
    // parse configure file
    constexpr double MinValue = std::numeric_limits<double>::min();
    constexpr double MaxValue = std::numeric_limits<double>::max();

    struct
    {
        bool timing_only = false;
        bool robust_test = false;
    } args;

    Config config = Config {
        // Input
        "", // std::string tet_mesh_file
        "", // std::string func_file;
        "", // std::string output_dir;

        // Lookup setting.
        false, // bool use_lookup;
        false, // bool use_secondary_lookup;
        true, // bool use_topo_ray_shooting;

        // Parameter for tet grid generation.
        // (Only used if tet_mesh_file is empty.)

        // We use 1000k tetrahedrons, as performed in the article to assure an adequate good resolution
        99, // size_t            tet_mesh_resolution;

        { MaxValue, MaxValue, MaxValue }, // std::array<double, 3> tet_mesh_bbox_min;
        { MinValue, MinValue, MinValue }, // std::array<double, 3> tet_mesh_bbox_max;
    };

    // Generate bounding box from site set
    const std::size_t siteNb = sites.size / 4;
    for ( std::size_t i = 0; i < siteNb; i++ )
    {
        const apo::Real radius = sites[ i * 4 + 3 ];
        for ( std::size_t c = 0; c < 3; c++ )
        {
            config.tet_mesh_bbox_min[ c ] = std::min( config.tet_mesh_bbox_min[ c ], static_cast<double>( sites[ i * 4 + c ] - radius ) );
            config.tet_mesh_bbox_max[ c ] = std::max( config.tet_mesh_bbox_max[ c ], static_cast<double>( sites[ i * 4 + c ] + radius ) );
        }
    }

    if ( config.use_lookup )
    {
        // load lookup table
        std::cout << "load table ..." << std::endl;
        bool loaded = load_lookup_table( simplicial_arrangement::MATERIAL_INTERFACE );
        if ( loaded )
        {
            std::cout << "loading finished." << std::endl;
        }
        else
        {
            std::cout << "loading failed." << std::endl;
            throw std::runtime_error( "Can't find simplicial arrangement lookup table" );
        }
    }
    else
    {
        disable_lookup_table();
        config.use_secondary_lookup = false;
    }

    // load tet mesh
    std::vector<std::array<double, 3>> pts;
    std::vector<std::array<size_t, 4>> tets;
    if ( config.tet_mesh_file != "" )
    {
        std::cout << "load mesh file " << config.tet_mesh_file << std::endl;
        load_tet_mesh( config.tet_mesh_file, pts, tets );
    }
    else
    {
        std::cout << "generating mesh with resolution " << config.tet_mesh_resolution << std::endl;
        generate_tet_mesh( config.tet_mesh_resolution, config.tet_mesh_bbox_min, config.tet_mesh_bbox_max, pts, tets );
    }

    // load implicit functions and compute function values at vertices
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> funcVals;

    // We do not rely on functions file since sites are provided dynamically
    // if ( load_functions( config.func_file, pts, funcVals ) )
    // {
    //     std::cout << "function loading finished." << std::endl;
    // }
    // else
    // {
    //     std::cout << "function loading failed." << std::endl;
    //     throw std::runtime_error( "function loading failed" );
    // }

    // Hardcoded sphere functions for Apollonius validation
    auto n_pts  = static_cast<Eigen::Index>( pts.size() );
    auto n_func = static_cast<Eigen::Index>( siteNb );
    funcVals.resize( n_pts, n_func );
    for ( int j = 0; j < n_func; ++j )
    {
        std::array<double, 3> center;
        for ( int i = 0; i < 3; ++i )
            center[ i ] = static_cast<double>( sites[ j * 4 + i ] );

        auto radius = static_cast<double>( sites[ j * 4 + 3 ] );
        //
        std::unique_ptr<ImplicitFunction<double>> sphere;
        if ( radius >= 0 )
        {
            sphere = std::make_unique<SphereDistanceFunction<double>>( center, radius );
        }
        else
        {
            sphere = std::make_unique<SphereUnsignedDistanceFunction<double>>( center, -radius );
        }
        for ( int i = 0; i < n_pts; i++ )
        {
            funcVals( i, j ) = sphere->evaluate( pts[ i ][ 0 ], pts[ i ][ 1 ], pts[ i ][ 2 ] );
        }
    }

    // compute implicit arrangement
    std::vector<std::array<double, 3>>     MI_pts;
    std::vector<PolygonFace>               MI_faces;
    std::vector<std::vector<size_t>>       patches;
    std::vector<std::pair<size_t, size_t>> patch_function_label;
    std::vector<Edge>                      MI_edges;
    std::vector<std::vector<size_t>>       chains;
    std::vector<std::vector<size_t>>       non_manifold_edges_of_vert;
    std::vector<std::vector<size_t>>       shells;
    std::vector<std::vector<size_t>>       material_cells;
    std::vector<size_t>                    cell_function_label;
    // record timings
    std::vector<std::string> timing_labels;
    std::vector<double>      timings;
    // record stats
    std::vector<std::string> stats_labels;
    std::vector<size_t>      stats;

    if ( !material_interface( args.robust_test,
                              config.use_lookup,
                              config.use_secondary_lookup,
                              config.use_topo_ray_shooting,
                              //
                              pts,
                              tets,
                              funcVals,
                              //
                              MI_pts,
                              MI_faces,
                              patches,
                              patch_function_label,
                              MI_edges,
                              chains,
                              non_manifold_edges_of_vert,
                              shells,
                              material_cells,
                              cell_function_label,
                              timing_labels,
                              timings,
                              stats_labels,
                              stats ) )
    {
        throw std::runtime_error( "Unable to compute material interface" );
    }

    if( !meshPath.empty() && !material_cells.empty() )
        save_result_MI(meshPath + "/mesh.json",
                       MI_pts,
                       MI_faces,
                       patches,
                       patch_function_label,
                       MI_edges,
                       chains,
                       non_manifold_edges_of_vert,
                       shells,
                       material_cells,
                       cell_function_label);

    std::vector<std::set<std::size_t>> pointsLabels {};
    pointsLabels.resize( MI_pts.size(), std::set<std::size_t> {} );

    for ( std::size_t p = 0; p < patches.size(); p++ )
    {
        const auto [ i, j ] = patch_function_label[ p ];
        for ( const std::size_t f : patches[ p ] )
        {
            const PolygonFace & face = MI_faces[ f ];
            for ( const std::size_t v : face.vert_indices )
            {
                pointsLabels[ v ].emplace( i );
                pointsLabels[ v ].emplace( j );
            }
        }
    }

    struct Bisector
    {
        uint32_t i, j;
        bool     operator<( const Bisector & b ) const { return i < b.i || ( i == b.i && j < b.j ); }
    };
    struct Trisector
    {
        uint32_t i, j, k;
        bool     operator<( const Trisector & b ) const
        {
            return i < b.i || ( i == b.i && j < b.j ) || ( i == b.i && j == b.j && k < b.k );
        }
    };
    struct Quadrisector
    {
        uint32_t i, j, k, l;
        bool     operator<( const Quadrisector & b ) const
        {
            return i < b.i || ( i == b.i && j < b.j ) || ( i == b.i && j == b.j && k < b.k )
                   || ( i == b.i && j == b.j && k == b.k && l < b.l );
        }
    };

    std::set<Bisector>     bisectors {};
    std::set<Trisector>    trisectors {};
    std::set<Quadrisector> quadrisectors {};
    for ( std::size_t v = 0; v < pointsLabels.size(); v++ )
    {
        const std::set<std::size_t> & labels = pointsLabels[ v ];
        if ( labels.empty() ) // point is not on a boundary
            continue;

        auto iterator = labels.begin();
        if ( labels.size() == 2 )
        {
            const uint32_t i = static_cast<uint32_t>( *iterator++ );
            const uint32_t j = static_cast<uint32_t>( *iterator );
            bisectors.emplace( Bisector { i, j } );
        }
        else if ( labels.size() == 3 )
        {
            const uint32_t i = static_cast<uint32_t>( *iterator++ );
            const uint32_t j = static_cast<uint32_t>( *iterator++ );
            const uint32_t k = static_cast<uint32_t>( *iterator );
            trisectors.emplace( Trisector { i, j, k } );
        }
        else if ( labels.size() == 4 )
        {
            const uint32_t i = static_cast<uint32_t>( *iterator++ );
            const uint32_t j = static_cast<uint32_t>( *iterator++ );
            const uint32_t k = static_cast<uint32_t>( *iterator++ );
            const uint32_t l = static_cast<uint32_t>( *iterator );
            quadrisectors.emplace( Quadrisector { i, j, k, l } );
        }
        else
        {
            throw std::runtime_error( "Found a point with more than 4 labels." );
        }
    }

    apo::gpu::Topology topology {};
    topology.bisectors.reserve( bisectors.size() * 2 );
    for ( const Bisector & bisector : bisectors )
    {
        topology.bisectors.emplace_back( bisector.i );
        topology.bisectors.emplace_back( bisector.j );
    }

    topology.trisectors.reserve( trisectors.size() * 3 );
    for ( const Trisector & trisector : trisectors )
    {
        topology.trisectors.emplace_back( trisector.i );
        topology.trisectors.emplace_back( trisector.j );
        topology.trisectors.emplace_back( trisector.k );
    }

    topology.quadrisectors.reserve( quadrisectors.size() * 4 );
    for ( const Quadrisector & quadrisector : quadrisectors )
    {
        topology.quadrisectors.emplace_back( quadrisector.i );
        topology.quadrisectors.emplace_back( quadrisector.j );
        topology.quadrisectors.emplace_back( quadrisector.k );
        topology.quadrisectors.emplace_back( quadrisector.l );
    }

    return topology;
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

        // Robust
        const std::string outFolder = fmt::format( "./topology-robust-{}", name );
        if( !std::filesystem::exists( outFolder ) )
            std::filesystem::create_directory( outFolder );

        const apo::gpu::Topology robustTopology = getRobustTopology( sites, outFolder );
        const std::vector<char>  robustOut      = apo::toCsv( robustTopology );

        auto robustOutputCsv      = std::ofstream { fmt::format( outFolder + "/topology-robust-{}.json", name ) };
        auto robustOutputIterator = std::ostream_iterator<char> { robustOutputCsv, "" };
        std::copy( robustOut.begin(), robustOut.end(), robustOutputIterator );
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

        // Robust
        const std::string outFolder = fmt::format( "./topology-robust-{}", pdb );
        if( !std::filesystem::exists( outFolder ) )
            std::filesystem::create_directory( outFolder );

        const apo::gpu::Topology robustTopology = getRobustTopology( sites, outFolder );
        const std::vector<char>  robustOut      = apo::toCsv( robustTopology );

        auto robustOutputCsv      = std::ofstream { fmt::format( outFolder + "/topology-robust-{}.json", pdb ) };
        auto robustOutputIterator = std::ostream_iterator<char> { robustOutputCsv, "" };
        std::copy( robustOut.begin(), robustOut.end(), robustOutputIterator );
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

        // Robust
        const std::string outFolder = fmt::format(
                "./topology-robust-{}-{}-{}-{}",
                configuration.count,
                configuration.spreading,
                configuration.radiiFactor,
                configuration.radiiStart
        );

        if( !std::filesystem::exists( outFolder ) )
            std::filesystem::create_directory( outFolder );

        const apo::gpu::Topology robustTopology = getRobustTopology( sites, outFolder );
        const std::vector<char>  robustOut      = apo::toCsv( robustTopology );

        auto robustOutputCsv = std::ofstream {
            fmt::format( outFolder + "/topology-robust-{}-{}-{}-{}.json",
                         configuration.count,
                         configuration.spreading,
                         configuration.radiiFactor,
                         configuration.radiiStart )
        };
        auto robustOutputIterator = std::ostream_iterator<char> { robustOutputCsv, "" };
        std::copy( robustOut.begin(), robustOut.end(), robustOutputIterator );
    }

    return EXIT_SUCCESS;
}
