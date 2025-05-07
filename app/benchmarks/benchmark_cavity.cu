#include <fstream>

#include <apo/core/utils.hpp>
#include <apo/gpu/benchmark.cuh>
#include <thrust/adjacent_difference.h>

#include "common/samples.hpp"
#include "optix/sphere_module.cuh"
#include "utils.hpp"

namespace apo::gpu
{
    std::vector<double> benchmarkVertices( const uint32_t  warmupNb,
                                           const uint32_t  sampleNb,
                                           ConstSpan<Real> sitesData,
                                           DeviceBuffer &  dVertices,
                                           uint32_t &      vertexNb )
    {
        return apo::Benchmark( "Apollonius vertices" )
            .timerFunction( apo::gpu::timer_ms )
            .warmups( warmupNb )
            .iterations( sampleNb )
            .printStats()
            .run( [ & ] { vertexNb = getVertices( sitesData, dVertices ); } );
    }

    std::vector<double> benchmarkHitRatios( const uint32_t   warmupNb,
                                            const uint32_t   sampleNb,
                                            const uint32_t   vertexNb,
                                            ConstSpan<float> sitesData,
                                            DeviceBuffer &   dVertices,
                                            DeviceBuffer &   dHitRatios )
    {
        // Configuration
        apo::optix::Context optixContext {};

        // Initialize OptiX pipelines
        apo::optix::GeometryPipeline pipeline { optixContext };
        pipeline.setPrimitiveType( OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE );

        apo::optix::Module rayGen { optixContext, "ptx/find_occlusion.ptx" };
        pipeline.setRayGen( rayGen, "__raygen__rg" );
        pipeline.setMiss( rayGen, "__miss__general" );

        apo::optix::SphereModule & sphereModule
            = pipeline.add( pipeline, "ptx/find_occlusion.ptx", "__closesthit__site" );
        apo::optix::SphereGeometry geometry { optixContext, sitesData };
        sphereModule.add( geometry );

        pipeline.compile();
        pipeline.updateGeometry();

        constexpr uint32_t OcclusionSampleNb = 64;
        auto               directions        = apo::gpu::getDirections( OcclusionSampleNb );

        dHitRatios = apo::gpu::DeviceBuffer::Typed<float>( vertexNb, true );

        // Trace
        apo::gpu::DeviceBuffer      dParameters = apo::gpu::DeviceBuffer::Typed<apo::OcclusionDetectionData>( 1 );
        apo::OcclusionDetectionData parameters {};
        parameters.handle     = pipeline.getHandle();
        parameters.vertexNb   = vertexNb;
        parameters.vertices   = dVertices.get<float4>();
        parameters.sampleNb   = OcclusionSampleNb;
        parameters.directions = directions.get<float3>();
        parameters.hitRatios  = dHitRatios.get<float>();

        const auto sbt = pipeline.getBindingTable();
        apo::gpu::cudaCheck( cudaMemcpyAsync( dParameters.get(),
                                              &parameters,
                                              sizeof( apo::OcclusionDetectionData ),
                                              cudaMemcpyHostToDevice,
                                              optixContext.getStream() ) );

        return Benchmark( "Hit ratio" )
            .timerFunction( apo::gpu::timer_ms )
            .warmups( warmupNb )
            .iterations( sampleNb )
            .printStats()
            .run(
                [ & ]
                {
                    pipeline.launch( dParameters.get(), sizeof( apo::OcclusionDetectionData ), sbt, vertexNb, 1, 1 );
                    ;
                } );
    }

    std::vector<double> benchmarkFiltering( const uint32_t           warmupNb,
                                            const uint32_t           sampleNb,
                                            const uint32_t           vertexNb,
                                            const float              hitRatioThreshold,
                                            const float              radiusThreshold,
                                            apo::gpu::DeviceBuffer & dVertices,
                                            apo::gpu::DeviceBuffer & dHitRatios )
    {
        return Benchmark( "Hit ratio" )
            .timerFunction( apo::gpu::timer_ms )
            .warmups( warmupNb )
            .iterations( sampleNb )
            .printStats()
            .run( [ & ] { getVertices( vertexNb, dVertices, dHitRatios, hitRatioThreshold, radiusThreshold ); } );
    }
} // namespace apo::gpu

int main( int, char ** )
{
    constexpr uint32_t         SampleNb       = 100;
    constexpr uint32_t         WarmupNb       = 10;
    constexpr std::string_view Configuration  = "I9-13900K-RTX-4090";
    float                      filterRadius   = 5.f;
    float                      filterHitRatio = .5f;

    const std::vector<apo::Path> proteinTestSet = {
        "samples/7P3W.mmtf", // 37149
        "samples/4V8W.mmtf", // 123082
        "samples/7LER.mmtf", // 158430
        "samples/6RXU.mmtf", // 211834
        "samples/4V6X.mmtf", // 237685
    };

    std::vector<char> out {};
    fmt::format_to( std::back_inserter( out ),
                    "Molecule;SiteCount;VertexCount;Iteration;VertexTime;HitRatiosTime;FilteringTime\n" );

    for ( const apo::Path & path : proteinTestSet )
    {
        std::vector<apo::Real> sites  = apo::loadProtein( path );
        const std::size_t      siteNb = sites.size() / 4;
        apo::logger::info( "Perform benchmark for {} with {} sites", path, siteNb );

        // Benchmark vertices computation
        apo::gpu::DeviceBuffer dVertices;
        uint32_t               vertexNb = 0;
        std::vector<double>    verticesSamples
            = apo::gpu::benchmarkVertices( WarmupNb, SampleNb, sites, dVertices, vertexNb );

        // Sites to float for tracing
        std::vector<float> fSites {};
        fSites.reserve( sites.size() );
        for ( const apo::Real v : sites )
            fSites.emplace_back( v );

        // Benchmark hit ratio
        apo::gpu::DeviceBuffer dHitRatios;
        std::vector<double>    hitRatiosSamples
            = benchmarkHitRatios( WarmupNb, SampleNb, vertexNb, fSites, dVertices, dHitRatios );

        // Benchmark filtering
        std::vector<double> filteringSamples
            = benchmarkFiltering( WarmupNb, SampleNb, vertexNb, filterHitRatio, filterRadius, dVertices, dHitRatios );

        // Output benchmarks
        const std::string pdb = std::filesystem::path( path ).stem().string();
        for ( std::size_t j = 0; j < SampleNb; j++ )
            fmt::format_to( std::back_inserter( out ),
                            "{};{};{};{};{};{};{}\n",
                            pdb,
                            siteNb,
                            vertexNb,
                            j,
                            verticesSamples[ j ],
                            hitRatiosSamples[ j ],
                            filteringSamples[ j ] );
    }

    auto outputCsv      = std::ofstream { fmt::format(
        "./benchmark-cavity-proteins-warm-{}-samples-{}-{}.csv", WarmupNb, SampleNb, Configuration ) };
    auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
    std::copy( out.begin(), out.end(), outputIterator );

    return EXIT_SUCCESS;
}