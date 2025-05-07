#include "optix/geometry_pipeline.cuh"

namespace apo::optix
{
    inline void GeometryPipeline::setRayGen( Module & oModule, std::string name )
    {
        oModule.compile( *this );
        m_rayGenModule = std::make_pair( &oModule, name );
    }

    inline void GeometryPipeline::setMiss( Module & oModule, std::string name )
    {
        oModule.compile( *this );
        m_missModule = std::make_pair( &oModule, name );
    }

    inline OptixTraversableHandle GeometryPipeline::getHandle() const { return m_handle; }
} // namespace apo::optix
