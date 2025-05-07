#include <fstream>

#include <apo/gpu/algorithm.cuh>
#include <apo/gpu/benchmark.cuh>

#include "common/samples.hpp"

namespace apo::gpu
{
    std::vector<AlgorithmGPU<>::Sample> stagesBenchmark( uint32_t             warmupNb,
                                                         uint32_t             sampleNb,
                                                         ConstSpan<apo::Real> sites )
    {
        AlgorithmGPU algorithm { sites };
        for ( uint32_t i = 0; i < warmupNb; i++ )
            algorithm.build();

        std::vector<AlgorithmGPU<>::Sample> samples {};
        samples.reserve( sampleNb );
        for ( uint32_t i = 0; i < sampleNb; i++ )
        {
            algorithm.build();
            samples.emplace_back( algorithm.sample );
        }

        return samples;
    }
} // namespace apo::gpu

int main( int, char ** )
{
    constexpr uint32_t         SampleNb      = 100;
    constexpr uint32_t         WarmupNb      = 10;
    constexpr std::string_view Configuration = "I9-13900K-RTX-4090";

    {
        const std::vector<apo::Path> proteinTestSet = {
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

        std::vector<char> out {};
        fmt::format_to( std::back_inserter( out ),
                        "Molecule;SiteCount;Iteration;initializationTime;bvhConstructionTime;knnConstructionTime;"
                        "knnSearchTime;vertexValidationTime;edgeValidationTime;bisectorValidationTime\n" );
        for ( const apo::Path & path : proteinTestSet )
        {
            std::vector<apo::Real> sites  = apo::loadProtein( path );
            const std::size_t      siteNb = sites.size() / 4;
            const std::string      pdb    = std::filesystem::path( path ).stem().string();
            apo::logger::info( "Perform benchmark for {} with {} sites", pdb, siteNb );

            // Random perturbation to ensure general position
            apo::joggle( sites, 1e-3 );

            // Benchmark apo
            std::vector<apo::gpu::AlgorithmGPU<>::Sample> apoSamples
                = apo::gpu::stagesBenchmark( WarmupNb, SampleNb, sites );

            // output apo
            for ( std::size_t j = 0; j < apoSamples.size(); j++ )
            {
                fmt::format_to( std::back_inserter( out ),
                                "{};{};{};{};{};{};{};{};{};{}\n",
                                pdb,
                                siteNb,
                                j,
                                apoSamples[ j ].initialization,
                                apoSamples[ j ].bvhConstruction,
                                apoSamples[ j ].knnConstruction,
                                apoSamples[ j ].knnSearch,
                                apoSamples[ j ].vertexValidation,
                                apoSamples[ j ].edgeValidation,
                                apoSamples[ j ].bisectorValidation );
            }
        }

        auto outputCsv      = std::ofstream { fmt::format(
            "./apo-stages-proteins-warm-{}-samples-{}-{}.csv", WarmupNb, SampleNb, Configuration ) };
        auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
        std::copy( out.begin(), out.end(), outputIterator );
    }

    {
        struct CloudConfiguration
        {
            uint32_t  count;
            apo::Real spreading;
            apo::Real radiiFactor;
            apo::Real radiiStart;
        };

        std::vector<CloudConfiguration> cloudTestSet = {
            CloudConfiguration { 100, 25., 2., 1. },      CloudConfiguration { 1000, 50., 2., 1. },
            CloudConfiguration { 10000, 250., 2., 1. },   CloudConfiguration { 100000, 500., 2., 1. },

            CloudConfiguration { 100, 100., 10., 0.1 },   CloudConfiguration { 1000, 200., 10., 0.1 },
            CloudConfiguration { 10000, 500., 10., 0.1 }, CloudConfiguration { 100000, 1000., 10., 0.1 },
        };

        std::vector<char> out {};
        fmt::format_to(
            std::back_inserter( out ),
            "SiteCount;Spreading;radiiFactor;radiiStart;Iteration;initializationTime;bvhConstructionTime;"
            "knnConstructionTime;knnSearchTime;vertexValidationTime;edgeValidationTime;bisectorValidationTime\n" );
        for ( const CloudConfiguration & configuration : cloudTestSet )
        {
            // Load cloud
            std::vector<apo::Real> sites = apo::getUniform(
                configuration.count, configuration.spreading, configuration.radiiFactor, configuration.radiiStart );
            apo::logger::info(
                "Perform benchmark for configuration of sites: {}, spreading: {}, radiiFactor: {}, radiiStart: {}",
                configuration.count,
                configuration.spreading,
                configuration.radiiFactor,
                configuration.radiiStart );

            // Random perturbation to ensure general position
            apo::joggle( sites, 1e-3 );

            // Benchmark apo
            std::vector<apo::gpu::AlgorithmGPU<>::Sample> apoSamples
                = apo::gpu::stagesBenchmark( WarmupNb, SampleNb, sites );

            // output apo
            for ( std::size_t j = 0; j < apoSamples.size(); j++ )
                fmt::format_to( std::back_inserter( out ),
                                "{};{};{};{};{};{};{};{};{};{};{};{}\n",
                                configuration.count,
                                configuration.spreading,
                                configuration.radiiFactor,
                                configuration.radiiStart,
                                j,
                                apoSamples[ j ].initialization,
                                apoSamples[ j ].bvhConstruction,
                                apoSamples[ j ].knnConstruction,
                                apoSamples[ j ].knnSearch,
                                apoSamples[ j ].vertexValidation,
                                apoSamples[ j ].edgeValidation,
                                apoSamples[ j ].bisectorValidation );
        }

        auto outputCsv      = std::ofstream { fmt::format(
            "./apo-stages-clouds-warm-{}-samples-{}-{}.csv", WarmupNb, SampleNb, Configuration ) };
        auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
        std::copy( out.begin(), out.end(), outputIterator );
    }

    return EXIT_SUCCESS;
}