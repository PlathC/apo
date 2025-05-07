#ifndef APO_OPTIX_GEOMETRY_CUH
#define APO_OPTIX_GEOMETRY_CUH

#include "optix/context.cuh"

namespace apo::optix
{
    class BaseGeometry
    {
      public:
        BaseGeometry() = default;
        BaseGeometry( const Context & context );

        BaseGeometry( const BaseGeometry & )             = delete;
        BaseGeometry & operator=( const BaseGeometry & ) = delete;

        BaseGeometry( BaseGeometry && )             = default;
        BaseGeometry & operator=( BaseGeometry && ) = default;

        virtual ~BaseGeometry() = default;

        virtual void                  build()               = 0;
        virtual uint32_t              getGeometryNb() const = 0;
        inline OptixTraversableHandle getGASHandle() const;

      protected:
        const Context *        m_context;
        OptixTraversableHandle m_gasHandle;
    };
} // namespace apo::optix

#include "optix/geometry.inl"

#endif // APO_OPTIX_GEOMETRY_CUH