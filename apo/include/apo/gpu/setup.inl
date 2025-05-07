#include <fmt/printf.h>

#include "apo/gpu/setup.hpp"

namespace apo::gpu
{
#ifdef __CUDACC__
    inline void cudaCheck( std::string_view msg ) { cudaCheck( msg, cudaGetLastError() ); }

    inline void cudaCheck( std::string_view msg, cudaError_t err )
    {
        if ( err != cudaSuccess )
            fmt::print( "{}: {}\n", msg, cudaGetErrorString( err ) );
    }

    inline void cudaCheck( cudaError_t err )
    {
        if ( err != cudaSuccess )
            fmt::print( "{}\n", cudaGetErrorString( err ) );
    }

    inline KernelConfig KernelConfig::From( uint32_t n, uint32_t blockDim ) { return { n / blockDim + 1, blockDim }; }

    inline KernelConfig::operator std::tuple<dim3, dim3>() const { return { gridDim, blockDim }; }
#endif // __CUDACC__
} // namespace apo::gpu