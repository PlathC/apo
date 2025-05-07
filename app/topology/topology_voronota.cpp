#include <fstream>

#include <apo/core/utils.hpp>
#include <apo/gpu/algorithm.hpp>
#include <common/samples.hpp>
#include <common/voronota.hpp>

#include "topology.hpp"

apo::gpu::Topology getVoronotaTopology( voronota::ParallelComputationResult & result )
{
    // Merge quadruples maps
    for ( std::size_t i = 0; i < result.distributed_quadruples_maps.size(); i++ )
    {
        result.number_of_produced_quadruples += result.distributed_quadruples_maps[ i ].size();
        voronota::apollota::Triangulation::merge_quadruples_maps( result.distributed_quadruples_maps[ i ],
                                                                  result.merged_quadruples_map );
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

    for ( const auto & value : result.merged_quadruples_map )
    {
        const voronota::apollota::Quadruple & quadruple = value.first;
        quadrisectors.emplace( Quadrisector {
            static_cast<uint32_t>( quadruple.get( 0 ) ),
            static_cast<uint32_t>( quadruple.get( 1 ) ),
            static_cast<uint32_t>( quadruple.get( 2 ) ),
            static_cast<uint32_t>( quadruple.get( 3 ) ),
        } );

        for ( uint32_t i = 0; i < quadruple.size() - 1; i++ )
            for ( uint32_t j = i + 1; j < quadruple.size(); j++ )
                bisectors.emplace( Bisector { static_cast<uint32_t>( quadruple.get( i ) ),
                                              static_cast<uint32_t>( quadruple.get( j ) ) } );

        for ( uint32_t i = 0; i < quadruple.size() - 2; i++ )
            for ( uint32_t j = i + 1; j < quadruple.size() - 1; j++ )
                for ( uint32_t k = j + 1; k < quadruple.size(); k++ )
                    trisectors.emplace( Trisector { static_cast<uint32_t>( quadruple.get( i ) ),
                                                    static_cast<uint32_t>( quadruple.get( j ) ),
                                                    static_cast<uint32_t>( quadruple.get( k ) ) } );
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
        const apo::gpu::Topology voroTopology = getVoronotaTopology( result );
        const std::vector<char>  voroOut      = apo::toCsv( voroTopology );

        auto voroOutputCsv      = std::ofstream { fmt::format( "./topology-voronota-{}.json", name ) };
        auto voroOutputIterator = std::ostream_iterator<char> { voroOutputCsv, "" };
        std::copy( voroOut.begin(), voroOut.end(), voroOutputIterator );
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
        const apo::gpu::Topology voroTopology = getVoronotaTopology( result );
        const std::vector<char>  voroOut      = apo::toCsv( voroTopology );

        auto voroOutputCsv      = std::ofstream { fmt::format( "./topology-voronota-{}.json", pdb ) };
        auto voroOutputIterator = std::ostream_iterator<char> { voroOutputCsv, "" };
        std::copy( voroOut.begin(), voroOut.end(), voroOutputIterator );
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
        const apo::gpu::Topology voroTopology = getVoronotaTopology( result );
        const std::vector<char>  voroOut      = apo::toCsv( voroTopology );

        auto voroOutputCsv = std::ofstream {
            fmt::format( "./topology-voronota-{}-{}-{}-{}.json",
                         configuration.count,
                         configuration.spreading,
                         configuration.radiiFactor,
                         configuration.radiiStart )
        };
        auto voroOutputIterator = std::ostream_iterator<char> { voroOutputCsv, "" };
        std::copy( voroOut.begin(), voroOut.end(), voroOutputIterator );
    }

    return EXIT_SUCCESS;
}