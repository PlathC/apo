#ifndef APO_GPU_SETUP_HPP
#define APO_GPU_SETUP_HPP

#include <string_view>
#include <tuple>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>

enum cudaError;
using cudaError_t = cudaError;
typedef struct cudaGraphicsResource * cudaGraphicsResource_t;
enum cudaGraphicsRegisterFlags;
#endif // __CUDACC__

#ifdef __CUDACC__
#define APO_HOST __host__
#define APO_DEVICE __device__
#else
#define APO_HOST
#define APO_DEVICE
#endif // __CUDACC__

namespace apo::gpu
{
#ifdef __CUDACC__
    inline void cudaCheck( std::string_view msg = "" );
    inline void cudaCheck( std::string_view msg, cudaError_t err );
    inline void cudaCheck( cudaError_t err );

    struct KernelConfig
    {
        inline static KernelConfig From( uint32_t n, uint32_t blockDim );

        dim3 gridDim  = { 1u, 1u, 1u };
        dim3 blockDim = { 1u, 1u, 1u };

        inline operator std::tuple<dim3, dim3>() const;
    };
#endif // __CUDACC__

    template<class Type>
    APO_DEVICE APO_HOST inline void sswap( Type & a, Type & b )
    {
        Type c = std::move( a );
        a      = std::move( b );
        b      = std::move( c );
    }
} // namespace apo::gpu

#include "apo/gpu/setup.inl"

#endif // APO_GPU_SETUP_HPP