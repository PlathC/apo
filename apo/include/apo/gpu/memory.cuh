#ifndef APO_GPU_MEMORY_CUH
#define APO_GPU_MEMORY_CUH

#include "apo/core/type.hpp"
#include "apo/gpu/setup.hpp"

namespace apo::gpu
{
    enum class MemcpyType
    {
        DeviceToHost,
        HostToDevice
    };

    template<MemcpyType type, class Type>
    void mmemcpy( Type * const dst, const Type * const src, std::size_t count );

    template<class Type>
    void copy( Type * const dst, const Type * const src, uint32_t count );

    class DeviceBuffer
    {
      public:
        template<class Type>
        static DeviceBuffer Typed( const std::size_t count, bool zeroInit = false );
        template<class Type>
        static DeviceBuffer From( ConstSpan<Type> data );

        DeviceBuffer() = default;
        DeviceBuffer( const std::size_t size, bool zeroInit = false );

        DeviceBuffer( const DeviceBuffer & )             = delete;
        DeviceBuffer & operator=( const DeviceBuffer & ) = delete;

        DeviceBuffer( DeviceBuffer && other ) noexcept;
        DeviceBuffer & operator=( DeviceBuffer && other ) noexcept;
        ~DeviceBuffer();

        void reset();

        inline uint8_t *       get();
        inline const uint8_t * get() const;

        template<class Type>
        Type * get( std::size_t offset = 0 );
        template<class Type>
        const Type * get( std::size_t offset = 0 ) const;

        operator bool() const;

        template<class Type>
        std::size_t size() const;
        std::size_t size() const;

        template<class Type>
        std::vector<Type> toHost();

      private:
        bool        m_initialized = false;
        std::size_t m_size        = 0;
        uint8_t *   m_ptr         = nullptr;
    };

} // namespace apo::gpu

#include "apo/gpu/memory.inl"

#endif // APO_GPU_MEMORY_CUH