#include <cassert>

#include "apo/gpu/memory.cuh"

namespace apo::gpu
{
#ifdef __CUDACC__
    template<MemcpyType type, class Type>
    void mmemcpy( Type * const dst, const Type * const src, std::size_t count )
    {
        const std::size_t size = count * sizeof( Type );
        if constexpr ( type == MemcpyType::HostToDevice )
        {
            cudaCheck( cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice ) );
        }
        else if constexpr ( type == MemcpyType::DeviceToHost )
        {
            cudaCheck( cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost ) );
        }
    }

    template<class Type>
    __global__ void copyImpl( Type * const __restrict__ dst, const Type * const __restrict__ src, uint32_t count )
    {
        for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x )
        {
            dst[ i ] = src[ i ];
        }
    }

    template<class Type>
    void copy( Type * const __restrict__ dst, const Type * const __restrict__ src, uint32_t count )
    {
        auto [ numBlocks, numThreads ] = KernelConfig::From( count, 256 );
        copyImpl<<<numBlocks, numThreads>>>( dst, src, count );
        cudaCheck( "Device to Device copy failed" );
    }
#endif // __CUDACC__

    template<class Type>
    DeviceBuffer DeviceBuffer::Typed( const std::size_t count, bool zeroInit )
    {
        return { count * sizeof( Type ), zeroInit };
    }

    template<class Type>
    DeviceBuffer DeviceBuffer::From( ConstSpan<Type> data )
    {
        DeviceBuffer buffer = { data.size * sizeof( Type ), false };
        mmemcpy<MemcpyType::HostToDevice>( buffer.get<Type>(), data.ptr, data.size );
        return buffer;
    }

    inline uint8_t *       DeviceBuffer::get() { return m_ptr; }
    inline const uint8_t * DeviceBuffer::get() const { return m_ptr; }

    template<class Type>
    Type * DeviceBuffer::get( std::size_t offset )
    {
        return reinterpret_cast<Type *>( m_ptr + offset );
    }

    template<class Type>
    const Type * DeviceBuffer::get( std::size_t offset ) const
    {
        return reinterpret_cast<const Type *>( m_ptr + offset );
    }

    template<class Type>
    std::size_t DeviceBuffer::size() const
    {
        return m_size / sizeof( Type );
    }

    template<class Type>
    std::vector<Type> DeviceBuffer::toHost()
    {
        assert( m_size % sizeof( Type ) == 0 && "It seems that this type is not suitable." );

        const std::size_t hostBufferSize = m_size / sizeof( Type );
        std::vector<Type> buffer         = std::vector<Type>( hostBufferSize );

        mmemcpy<MemcpyType::DeviceToHost>( reinterpret_cast<uint8_t *>( buffer.data() ), m_ptr, m_size );
        return buffer;
    }
} // namespace apo::gpu