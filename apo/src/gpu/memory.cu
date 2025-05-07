#include "apo/gpu/memory.cuh"

namespace apo::gpu
{
    DeviceBuffer::DeviceBuffer( const std::size_t size, bool zeroInit ) : m_size( size )
    {
        cudaCheck( cudaMallocAsync( reinterpret_cast<void **>( &m_ptr ), size, 0 ) );
        if ( zeroInit )
            cudaCheck( cudaMemset( m_ptr, 0, size ) );

        m_initialized = true;
    }

    DeviceBuffer::DeviceBuffer( DeviceBuffer && other ) noexcept
    {
        std::swap( m_initialized, other.m_initialized );
        std::swap( m_size, other.m_size );
        std::swap( m_ptr, other.m_ptr );
    }

    DeviceBuffer & DeviceBuffer::operator=( DeviceBuffer && other ) noexcept
    {
        std::swap( m_initialized, other.m_initialized );
        std::swap( m_size, other.m_size );
        std::swap( m_ptr, other.m_ptr );

        return *this;
    }

    DeviceBuffer::~DeviceBuffer() { reset(); }

    void DeviceBuffer::reset()
    {
        if ( m_initialized )
        {
            cudaCheck( cudaFreeAsync( m_ptr, 0 ) );
            m_ptr         = nullptr;
            m_initialized = false;
        }
    }

    DeviceBuffer::operator bool() const { return m_initialized; }

    std::size_t DeviceBuffer::size() const { return m_size; }
} // namespace apo::gpu