#ifndef APO_OPTIX_SETUP_CUH
#define APO_OPTIX_SETUP_CUH

#include <apo/core/logger.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_host.h>
#include <optix_stubs.h>

#define optixCheck( call )                                                                                       \
    {                                                                                                            \
        OptixResult res = call;                                                                                  \
        if ( res != OPTIX_SUCCESS )                                                                              \
            apo::logger::debug( "[OPTIX]: {} failed with code {} (line {})", #call, uint32_t( res ), __LINE__ ); \
    }

#define optixCheckLog( call )                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        OptixResult  res                 = call;                                                                       \
        const size_t sizeof_log_returned = sizeOfLog;                                                                  \
        sizeOfLog                        = sizeof( log ); /* reset sizeof_log for future calls */                      \
        if ( res != OPTIX_SUCCESS )                                                                                    \
        {                                                                                                              \
            apo::logger::debug( "[OPTIX]: {} failed with code {} (line {})", #call, uint32_t( res ), __LINE__ );       \
            apo::logger::debug( "[OPTIX]: {} {}", log, ( sizeof_log_returned > sizeof( log ) ? "<TRUNCATED>" : "" ) ); \
        }                                                                                                              \
    } while ( 0 )

namespace apo::optix
{
    template<typename T>
    struct Record
    {
        alignas( OPTIX_SBT_RECORD_ALIGNMENT ) char header[ OPTIX_SBT_RECORD_HEADER_SIZE ];
        T data;
    };
} // namespace apo::optix

#endif // APO_OPTIX_SETUP_CUH
