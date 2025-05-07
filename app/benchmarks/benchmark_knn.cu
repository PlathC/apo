#include <fstream>

#include <apo/core/logger.hpp>
#include <apo/gpu/algorithm.cuh>
#include <apo/gpu/benchmark.cuh>
#include <common/samples.hpp>

int main( int argc, char ** argv )
{
    constexpr uint32_t         SampleNb      = 100;
    constexpr uint32_t         WarmupNb      = 10;
    constexpr std::string_view Configuration = "I9-13900K-RTX-4090";
    constexpr uint32_t         MaxKnnSample  = 32;

    // Proteins benchmarks
    std::vector<char> out {};
    fmt::format_to( std::back_inserter( out ), "Molecule;SiteCount;KNNNb;Iteration;Time\n" );

    // Load protein
    apo::Path              proteinPath = "samples/1AON.mmtf"; // 58870;
    std::vector<apo::Real> sites       = apo::loadProtein( proteinPath );

    // Random perturbation to ensure general position
    apo::joggle( sites, 1e-3 );

    for ( uint32_t knnSample = 0; knnSample < MaxKnnSample; knnSample++ )
    {
        const std::size_t siteNb = sites.size() / 4;
        apo::logger::info( "Perform benchmark for {} with {} sites", proteinPath, siteNb );

        // Benchmark apo
        std::vector<double> apoSamples = apo::Benchmark( fmt::format( "apo with {} knn", knnSample ) )
                                             .timerFunction( apo::gpu::timer_ms )
                                             .warmups( WarmupNb )
                                             .iterations( SampleNb )
                                             .printStats()
                                             .run(
                                                 [ & ]
                                                 {
                                                     apo::gpu::AlgorithmGPU<1> algorithm { sites, knnSample };
                                                     algorithm.build();
                                                 } );

        // output apo
        const std::string pdb = std::filesystem::path( proteinPath ).stem().string();
        for ( std::size_t j = 0; j < apoSamples.size(); j++ )
            fmt::format_to( std::back_inserter( out ), "{};{};{};{};{}\n", pdb, siteNb, knnSample, j, apoSamples[ j ] );
    }

    auto outputCsv      = std::ofstream { fmt::format(
        "./benchmark-knn-proteins-warm-{}-samples-{}-{}.csv", WarmupNb, SampleNb, Configuration ) };
    auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
    std::copy( out.begin(), out.end(), outputIterator );

    // Clouds benchmarks
    struct CloudConfiguration
    {
        uint32_t  count;
        apo::Real spreading;
        apo::Real radiiFactor;
        apo::Real radiiStart;
    };

    // Load cloud
    CloudConfiguration cloudConfiguration { 10000, 500., 10., 0.1 };
    sites = apo::getUniform( cloudConfiguration.count,
                             cloudConfiguration.spreading,
                             cloudConfiguration.radiiFactor,
                             cloudConfiguration.radiiStart );
    apo::logger::info(
        "Perform benchmark for configuration of sites: {}, spreading: {}, radiiFactor: {}, radiiStart: {}",
        cloudConfiguration.count,
        cloudConfiguration.spreading,
        cloudConfiguration.radiiFactor,
        cloudConfiguration.radiiStart );

    // Random perturbation to ensure general position
    apo::joggle( sites, 1e-3 );

    out = std::vector<char> {};
    fmt::format_to( std::back_inserter( out ), "SiteCount;Spreading;radiiFactor;radiiStart;KNNNb;Iteration;Time\n" );
    for ( uint32_t knnSample = 0; knnSample < MaxKnnSample; knnSample++ )
    {
        // Benchmark apo
        std::vector<double> apoSamples = apo::Benchmark( fmt::format( "apo with {} knn", knnSample ) )
                                             .timerFunction( apo::gpu::timer_ms )
                                             .warmups( WarmupNb )
                                             .iterations( SampleNb )
                                             .printStats()
                                             .run(
                                                 [ & ]
                                                 {
                                                     apo::gpu::AlgorithmGPU<1> algorithm { sites, knnSample };
                                                     algorithm.build();
                                                 } );

        // output apo
        for ( std::size_t j = 0; j < apoSamples.size(); j++ )
            fmt::format_to( std::back_inserter( out ),
                            "{};{};{};{};{};{};{}\n",
                            cloudConfiguration.count,
                            cloudConfiguration.spreading,
                            cloudConfiguration.radiiFactor,
                            cloudConfiguration.radiiStart,
                            knnSample,
                            j,
                            apoSamples[ j ] );
    }

    outputCsv      = std::ofstream { fmt::format(
        "./benchmark-knn-clouds-warm-{}-samples-{}-{}.csv", WarmupNb, SampleNb, Configuration ) };
    outputIterator = std::ostream_iterator<char> { outputCsv, "" };
    std::copy( out.begin(), out.end(), outputIterator );
}