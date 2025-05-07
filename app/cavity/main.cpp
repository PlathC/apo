#include <fstream>

#include <polyscope/point_cloud.h>

#include "common/samples.hpp"
#include "utils.hpp"

int main( int, char ** )
{
    std::vector<apo::Real> protein = apo::loadProtein( "samples/1AON.mmtf" );
    const std::size_t      siteNb  = protein.size() / 4;

    // Sites to float for tracing
    std::vector<float> fProtein {};
    fProtein.reserve( protein.size() );
    for ( apo::Real v : protein )
        fProtein.emplace_back( v );

    // Random perturbation to ensure general position
    apo::joggle( protein, 1e-3 );

    // Precomputation
    apo::logger::info( "Compute Apollonius diagram for {} sites", siteNb );
    apo::gpu::DeviceBuffer dVertices;
    const uint32_t         totalVertexNb = apo::gpu::getVertices( protein, dVertices );

    apo::logger::info( "Compute hit ratios" );
    apo::gpu::DeviceBuffer dHitRatios = apo::gpu::getHitRatios( totalVertexNb, dVertices, fProtein );

    // Display
    polyscope::init();

    // Configuration
    polyscope::view::upDir              = polyscope::UpDir::ZUp;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::Tile;
    {
        std::vector<glm::vec3> sites {};
        std::vector<float>     sitesRadii {};
        std::vector<uint32_t>  sitesId {};
        sitesRadii.reserve( fProtein.size() );
        sitesId.reserve( fProtein.size() );
        for ( std::size_t i = 0; i < fProtein.size() / 4; i++ )
        {
            sites.emplace_back( fProtein[ i * 4 + 0 ], fProtein[ i * 4 + 1 ], fProtein[ i * 4 + 2 ] );
            sitesRadii.emplace_back( fProtein[ i * 4 + 3 ] );
            sitesId.emplace_back( static_cast<uint32_t>( i ) );
        }

        auto *     sitesCloud         = polyscope::registerPointCloud( "Sites", sites );
        const auto sitesRadiiQuantity = sitesCloud->addScalarQuantity( "Radius", sitesRadii );
        sitesCloud->setPointRadiusQuantity( sitesRadiiQuantity, false );
        sitesCloud->addScalarQuantity( "Id", sitesId );
    }

    float hitRatioThreshold        = .5f;
    float radiusThreshold          = 5.f;
    bool  startup                  = true;
    polyscope::state::userCallback = [ & ]()
    {
        ImGui::PushItemWidth( 100 );

        bool changed = ImGui::SliderFloat( "Hit ratio threshold", &hitRatioThreshold, 0., 1.f );
        changed |= ImGui::SliderFloat( "Radius threshold", &radiusThreshold, 0.f, 100.f );

        if ( changed || startup )
        {
            startup = false;

            const auto vertices
                = apo::gpu::getVertices( totalVertexNb, dVertices, dHitRatios, hitRatioThreshold, radiusThreshold );
            std::vector<glm::vec3> verticesPosition {};
            std::vector<float>     verticesRadii {};
            verticesPosition.reserve( vertices.size() );
            verticesRadii.reserve( vertices.size() );
            for ( std::size_t i = 0; i < vertices.size() / 4; i++ )
            {
                verticesPosition.emplace_back( vertices[ i * 4 + 0 ], vertices[ i * 4 + 1 ], vertices[ i * 4 + 2 ] );
                verticesRadii.emplace_back( vertices[ i * 4 + 3 ] );
            }

            auto *     verticesCloud         = polyscope::registerPointCloud( "Vertices", verticesPosition );
            const auto verticesRadiiQuantity = verticesCloud->addScalarQuantity( "Radius", verticesRadii );
            verticesCloud->setPointRadiusQuantity( verticesRadiiQuantity, false );
        }
    };
    polyscope::show();

    return EXIT_SUCCESS;
}