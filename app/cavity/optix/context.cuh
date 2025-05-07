#ifndef APO_OPTIX_CONTEXT_CUH
#define APO_OPTIX_CONTEXT_CUH

#include "optix/setup.cuh"

namespace apo::optix
{
    class Context
    {
      public:
        Context();

        Context( const Context & )           = delete;
        Context operator=( const Context & ) = delete;

        Context( Context && other ) noexcept;
        Context & operator=( Context && other ) noexcept;

        ~Context();

        inline CUstream           getStream() const;
        inline OptixDeviceContext getOptiXContext() const;

      private:
        CUstream           m_stream = 0;
        cudaDeviceProp     m_deviceProps;
        OptixDeviceContext m_optiXContext;
    };
} // namespace apo::optix

#include "optix/context.inl"

#endif // APO_OPTIX_CONTEXT_CUH