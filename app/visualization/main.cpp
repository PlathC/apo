#include <apo/gpu/algorithm.hpp>
#include <polyscope/curve_network.h>
#include <polyscope/point_cloud.h>

#include "common/samples.hpp"
#include "common/voronota.hpp"

int main( int, char ** )
{
    constexpr bool Restricted        = true;
    constexpr auto RestrictionRadius = apo::Real( 1 );

    std::vector<apo::Real> sites  = apo::loadProtein( "samples/1AON.mmtf" );
    const std::size_t      siteNb = sites.size() / 4;

    // Random perturbation to ensure general position
    apo::joggle( sites, 1e-3 );

    voronota::ParallelComputationResult result;
    result.input_spheres.resize( siteNb );
    for ( std::size_t s = 0; s < result.input_spheres.size(); s++ )
    {
        result.input_spheres[ s ].x = sites[ s * 4 + 0 ];
        result.input_spheres[ s ].y = sites[ s * 4 + 1 ];
        result.input_spheres[ s ].z = sites[ s * 4 + 2 ];
        result.input_spheres[ s ].r = sites[ s * 4 + 3 ];
    }

    // Computation
    constexpr bool WithVoronota = false;
    apo::logger::info( "Computing diagram for {} sites.", siteNb );
    if ( WithVoronota )
        apo::logger::info( "With Voronota." );
    else
        apo::logger::info( "With apo." );

    auto                    start   = std::chrono::system_clock::now();
    apo::gpu::VertexDiagram diagram = {};
    if ( !WithVoronota )
        diagram = apo::gpu::computeApolloniusDiagram( sites );

    // Voronota test
    if ( WithVoronota )
        voronota::ParallelComputationProcessingWithOpenMP::process( result );

    auto end          = std::chrono::system_clock::now();
    auto elapsed      = end - start;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>( elapsed );

    if ( WithVoronota )
    {
        for ( std::size_t i = 0; i < result.distributed_quadruples_maps.size(); i++ )
        {
            result.number_of_produced_quadruples += result.distributed_quadruples_maps[ i ].size();
            voronota::apollota::Triangulation::merge_quadruples_maps( result.distributed_quadruples_maps[ i ],
                                                                      result.merged_quadruples_map );
        }

        for ( const auto & value : result.merged_quadruples_map )
        {
            const voronota::apollota::Quadruple & quadruple = value.first;
            for ( auto & sphere : value.second )
            {
                diagram.vertices.emplace_back( sphere.x );
                diagram.vertices.emplace_back( sphere.y );
                diagram.vertices.emplace_back( sphere.z );
                diagram.vertices.emplace_back( sphere.r );
                diagram.verticesId.emplace_back( quadruple.get( 0 ) );
                diagram.verticesId.emplace_back( quadruple.get( 1 ) );
                diagram.verticesId.emplace_back( quadruple.get( 2 ) );
                diagram.verticesId.emplace_back( quadruple.get( 2 ) );
            }
        }
    }

    const std::size_t vertexNb = diagram.vertices.size() / 4;
    apo::logger::info( "Found {} vertices in {} ms.", vertexNb, milliseconds.count() );

    // Compute visualization bounding aabb
    glm::vec3 mmin = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    glm::vec3 mmax = {
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
    };
    if constexpr ( Restricted )
    {
        for ( uint32_t i = 0; i < siteNb; i++ )
        {
            const glm::vec4 site = {
                static_cast<float>( sites[ i * 4 + 0 ] ),
                static_cast<float>( sites[ i * 4 + 1 ] ),
                static_cast<float>( sites[ i * 4 + 2 ] ),
                static_cast<float>( sites[ i * 4 + 3 ] ),
            };

            mmin = glm::min( glm::vec3( site ) - site.w, mmin );
            mmax = glm::max( glm::vec3( site ) + site.w, mmax );
        }

        mmin -= RestrictionRadius;
        mmax += RestrictionRadius;
    }

    // Remove out of bonds vertices
    std::vector<glm::vec3> restrictedVertices {};
    std::vector<float>     restrictedVerticesRadii {};
    restrictedVertices.reserve( vertexNb );
    restrictedVerticesRadii.reserve( vertexNb );
    for ( std::size_t i = 0; i < vertexNb; i++ )
    {
        const glm::vec4 vertex = {
            static_cast<float>( diagram.vertices[ i * 4 + 0 ] ),
            static_cast<float>( diagram.vertices[ i * 4 + 1 ] ),
            static_cast<float>( diagram.vertices[ i * 4 + 2 ] ),
            static_cast<float>( diagram.vertices[ i * 4 + 3 ] ),
        };

        glm::vec3  center { vertex.x, vertex.y, vertex.z };
        const bool isInside = glm::all( glm::greaterThan( center - vertex.w, mmin ) )
                              && glm::all( glm::lessThan( center + vertex.w, mmax ) );
        if ( Restricted && !isInside )
            continue;

        restrictedVertices.emplace_back( vertex );
        restrictedVerticesRadii.emplace_back( vertex.w );
    }

    polyscope::init();

    // Configuration
    polyscope::view::upDir              = polyscope::UpDir::ZUp;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::Tile;

    // Sites
    {
        std::vector<glm::vec3> sitesPositions {};
        std::vector<float>     sitesRadii {};
        std::vector<uint32_t>  sitesId {};
        sitesRadii.reserve( siteNb );
        sitesId.reserve( siteNb );
        for ( uint32_t i = 0; i < siteNb; i++ )
        {
            const glm::vec4 site = {
                static_cast<float>( sites[ i * 4 + 0 ] ),
                static_cast<float>( sites[ i * 4 + 1 ] ),
                static_cast<float>( sites[ i * 4 + 2 ] ),
                static_cast<float>( sites[ i * 4 + 3 ] ),
            };

            sitesPositions.emplace_back( site );
            sitesRadii.emplace_back( site.w );
            sitesId.emplace_back( i );
        }

        auto *     sitesCloud         = polyscope::registerPointCloud( "Sites", sitesPositions );
        const auto sitesRadiiQuantity = sitesCloud->addScalarQuantity( "Radius", sitesRadii );
        sitesCloud->setPointRadiusQuantity( sitesRadiiQuantity, false );
        sitesCloud->addScalarQuantity( "Id", sitesId );
    }

    // Vertices
    polyscope::PointCloud * verticesCloud = polyscope::registerPointCloud( "Vertices", restrictedVertices );
    {
        const auto verticesRadiusQuantity = verticesCloud->addScalarQuantity( "Radius", restrictedVerticesRadii );
        verticesCloud->setPointRadiusQuantity( verticesRadiusQuantity, false );
    }

    if constexpr ( Restricted )
    {
        const std::array<glm::vec3, 8> aabbNodes {
            mmin,
            { mmax.x, mmin.y, mmin.z },
            { mmax.x, mmax.y, mmin.z },
            { mmin.x, mmax.y, mmin.z },
            { mmin.x, mmin.y, mmax.z },
            { mmax.x, mmin.y, mmax.z },
            mmax,
            { mmin.x, mmax.y, mmax.z },
        };
        const std::array<glm::ivec2, 12> aabbEdges {
            glm::ivec2 { 0, 1 }, glm::ivec2 { 1, 2 }, glm::ivec2 { 2, 3 }, glm::ivec2 { 3, 0 },
            glm::ivec2 { 0, 4 }, glm::ivec2 { 4, 5 }, glm::ivec2 { 1, 5 }, glm::ivec2 { 5, 6 },
            glm::ivec2 { 2, 6 }, glm::ivec2 { 6, 7 }, glm::ivec2 { 3, 7 }, glm::ivec2 { 7, 4 },
        };
        polyscope::registerCurveNetwork( "AABB", aabbNodes, aabbEdges );
    }

    polyscope::show();

    return EXIT_SUCCESS;
}
