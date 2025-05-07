#include <cstdio>
#include <fstream>
#include <numeric>
#include <optional>

#include <apo/core/logger.hpp>
#include <apo/gpu/algorithm.cuh>
#include <common/samples.hpp>
#include <cupti.h>

// Based on CUPTI samples
#define CUPTI_CALL( call )                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        CUptiResult _status = call;                                                                                    \
        if ( _status != CUPTI_SUCCESS )                                                                                \
        {                                                                                                              \
            const char * errstr;                                                                                       \
            cuptiGetResultString( _status, &errstr );                                                                  \
            fprintf( stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #call, errstr ); \
            exit( -1 );                                                                                                \
        }                                                                                                              \
    } while ( 0 )

#define BUF_SIZE ( 32 * 1024 )
#define ALIGN_SIZE ( 8 )
#define ALIGN_BUFFER( buffer, align )                                              \
    ( ( (uintptr_t)( buffer ) & ( (align)-1 ) )                                    \
          ? ( ( buffer ) + ( align ) - ( (uintptr_t)( buffer ) & ( (align)-1 ) ) ) \
          : ( buffer ) )

std::size_t deviceAllocationSize    = 0;
std::size_t deviceMaxAllocationSize = 0;

static void updateMemory( CUpti_Activity * record )
{
    switch ( record->kind )
    {
    case CUPTI_ACTIVITY_KIND_MEMORY2:
    {
        CUpti_ActivityMemory3 * memory = (CUpti_ActivityMemory3 *)(void *)record;
        if ( memory->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION )
        {
            deviceAllocationSize += memory->bytes;
        }
        else if ( memory->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE )
        {
            if ( deviceAllocationSize < memory->bytes )
                apo::logger::error( "More release than allocation." );
            deviceAllocationSize -= memory->bytes;
        }
        break;
    }
    default: break;
    }

    deviceMaxAllocationSize = std::max( deviceAllocationSize, deviceMaxAllocationSize );
}

void CUPTIAPI bufferRequested( uint8_t ** buffer, size_t * size, size_t * maxNumRecords )
{
    uint8_t * bfr = new uint8_t[ BUF_SIZE + ALIGN_SIZE ];

    if ( bfr == nullptr )
    {
        fmt::print( "Error: Out of memory.\n" );
        exit( -1 );
    }

    *size          = BUF_SIZE;
    *buffer        = ALIGN_BUFFER( bfr, ALIGN_SIZE );
    *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted( CUcontext ctx, uint32_t streamId, uint8_t * buffer, size_t size, size_t validSize )
{
    CUptiResult      status;
    CUpti_Activity * record = nullptr;

    if ( validSize > 0 )
    {
        do
        {
            status = cuptiActivityGetNextRecord( buffer, validSize, &record );
            if ( status == CUPTI_SUCCESS )
                updateMemory( record );
            else if ( status == CUPTI_ERROR_MAX_LIMIT_REACHED )
                break;
            else
                CUPTI_CALL( status );
        } while ( true );

        // report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL( cuptiActivityGetNumDroppedRecords( ctx, streamId, &dropped ) );
        if ( dropped != 0 )
            apo::logger::warning( "Warning: Dropped {} activity records.\n", dropped );
    }

    delete buffer;
}

void initializeCupti()
{
    CUPTI_CALL( cuptiActivityEnable( CUPTI_ACTIVITY_KIND_MEMORY2 ) );
    CUPTI_CALL( cuptiActivityEnable( CUPTI_ACTIVITY_KIND_MEMORY_POOL ) );

    // Register callbacks for buffer requests and for buffers completed by CUPTI.
    CUPTI_CALL( cuptiActivityRegisterCallbacks( bufferRequested, bufferCompleted ) );
}

void reset()
{
    deviceAllocationSize    = 0;
    deviceMaxAllocationSize = 0;
}

int64_t getMaxMemoryUse()
{
    // Force flush any remaining activity buffers before sending the max allocation size
    CUPTI_CALL( cuptiActivityFlushAll( 1 ) );

    return deviceMaxAllocationSize;
}

int64_t getPeakDeviceMemory( const std::function<void()> & task )
{
    reset();
    task();
    return getMaxMemoryUse() / 1e6;
}

int main( int, char ** )
{
    initializeCupti();

    {
        std::vector<apo::Path> proteinTestSet = {
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
        fmt::format_to( std::back_inserter( out ), "Molecule;SiteCount;MB\n" );

        for ( const apo::Path & path : proteinTestSet )
        {
            std::vector<apo::Real> sites  = apo::loadProtein( path );
            const std::size_t      siteNb = sites.size() / 4;
            const std::string      pdb    = std::filesystem::path( path ).stem().string();
            apo::logger::info( "Perform benchmark for {} with {} sites", pdb, siteNb );

            // Random perturbation to ensure general position
            apo::joggle( sites, 1e-3 );

            // Benchmark memory
            const int64_t mb = getPeakDeviceMemory(
                [ &sites ]()
                {
                    apo::gpu::AlgorithmGPU algorithm { sites };
                    algorithm.build();
                } );

            // output
            fmt::format_to( std::back_inserter( out ), "{};{};{}\n", pdb, siteNb, mb );
        }

        auto outputCsv      = std::ofstream { "./apo-proteins-memory.csv" };
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
        fmt::format_to( std::back_inserter( out ), "SiteCount;Spreading;radiiFactor;radiiStart;MB\n" );
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

            // Benchmark memory
            const int64_t mb = getPeakDeviceMemory(
                [ &sites ]()
                {
                    apo::gpu::AlgorithmGPU algorithm { sites };
                    algorithm.build();
                } );

            // output
            fmt::format_to( std::back_inserter( out ),
                            "{};{};{};{};{}\n",
                            configuration.count,
                            configuration.spreading,
                            configuration.radiiFactor,
                            configuration.radiiStart,
                            mb );
        }

        auto outputCsv      = std::ofstream { "./apo-clouds-memory.csv" };
        auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
        std::copy( out.begin(), out.end(), outputIterator );
    }

    return EXIT_SUCCESS;
}