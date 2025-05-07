#include <fstream>

#include <apo/core/utils.hpp>
#include <apo/gpu/algorithm.hpp>
#include <common/samples.hpp>

#include "precision.hpp"

struct PrecisionDiagramData
{
    std::vector<apo::precision::Quadrisector> quadrisectors {};
    std::vector<apo::precision::Trisector>    trisectors {};
};

PrecisionDiagramData getApoData( apo::ConstSpan<apo::Real> sites )
{
    apo::gpu::FullDiagram diagram = apo::gpu::computeFullApolloniusDiagram( sites );
    PrecisionDiagramData  data {};
    data.quadrisectors.reserve( diagram.vertices.size() * 2 );
    data.trisectors.reserve( diagram.closedEdgesMin.size() * 2 );

    const std::size_t vertexNb = diagram.vertices.size() / 4;
    for ( std::size_t v = 0; v < vertexNb; v++ )
    {
        data.quadrisectors.emplace_back( apo::precision::Quadrisector {
            diagram.verticesId[ v * 4 + 0 ],
            diagram.verticesId[ v * 4 + 1 ],
            diagram.verticesId[ v * 4 + 2 ],
            diagram.verticesId[ v * 4 + 3 ],
            diagram.vertices[ v * 4 + 0 ],
            diagram.vertices[ v * 4 + 1 ],
            diagram.vertices[ v * 4 + 2 ],
        } );
    }

    const std::size_t closedEdgeNb = diagram.closedEdgesMin.size() / 4;
    for ( std::size_t e = 0; e < closedEdgeNb; e++ )
    {
        data.trisectors.emplace_back( apo::precision::Trisector {
            diagram.closedEdgesId[ e * 4 + 0 ],
            diagram.closedEdgesId[ e * 4 + 1 ],
            diagram.closedEdgesId[ e * 4 + 2 ],
            diagram.closedEdgesMin[ e * 4 + 0 ],
            diagram.closedEdgesMin[ e * 4 + 1 ],
            diagram.closedEdgesMin[ e * 4 + 2 ],
        } );
    }

    return data;
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

        // APO
        const PrecisionDiagramData apoData   = getApoData( sites );
        const std::size_t          badQuadNb = apo::precision::getBadQuadrisectorNb( sites, apoData.quadrisectors );
        const std::size_t          badClosedTriNb = apo::precision::getBadTrisectorNb( sites, apoData.trisectors );

        fmt::format_to( std::back_inserter( out ),
                        "{};{};{};{};{};{}\n",
                        name,
                        siteNb,
                        badQuadNb,
                        apoData.quadrisectors.size(),
                        badClosedTriNb,
                        apoData.trisectors.size() );
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

        // APO
        const PrecisionDiagramData apoData   = getApoData( sites );
        const std::size_t          badQuadNb = apo::precision::getBadQuadrisectorNb( sites, apoData.quadrisectors );
        const std::size_t          badClosedTriNb = apo::precision::getBadTrisectorNb( sites, apoData.trisectors );

        fmt::format_to( std::back_inserter( out ),
                        "{};{};{};{};{};{}\n",
                        pdb,
                        siteNb,
                        badQuadNb,
                        apoData.quadrisectors.size(),
                        badClosedTriNb,
                        apoData.trisectors.size() );
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

        // APO
        const PrecisionDiagramData apoData   = getApoData( sites );
        const std::size_t          badQuadNb = apo::precision::getBadQuadrisectorNb( sites, apoData.quadrisectors );
        const std::size_t          badClosedTriNb = apo::precision::getBadTrisectorNb( sites, apoData.trisectors );

        fmt::format_to( std::back_inserter( out ),
                        "{}-{}-{}-{};{};{};{};{};{}\n",
                        configuration.count,
                        configuration.spreading,
                        configuration.radiiFactor,
                        configuration.radiiStart,
                        configuration.count,
                        badQuadNb,
                        apoData.quadrisectors.size(),
                        badClosedTriNb,
                        apoData.trisectors.size() );
    }

    auto outputCsv = std::ofstream { fmt::format( "./precision-apo{}.csv",
#if APO_REAL_SIZE == 4

                                                  "-fp32"
#else
                                                  ""
#endif // APO_REAL_SIZE
                                                  ) };
    auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
    std::copy( out.begin(), out.end(), outputIterator );

    return EXIT_SUCCESS;
}