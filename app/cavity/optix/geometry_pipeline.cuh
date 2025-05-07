#ifndef APO_OPTIX_GEOMETRY_PIPELINE_CUH
#define APO_OPTIX_GEOMETRY_PIPELINE_CUH

#include "apo/gpu/memory.cuh"
#include "optix/pipeline.cuh"
#include "optix/program.cuh"
#include "shaders/data.cuh"

namespace apo::optix
{
    class SphereModule;
    using GeometryHitGroupRecord = Record<GeometryHitGroup>;
    using ModuleFunction         = std::pair<Module *, std::string>;
    class GeometryPipeline : public Pipeline
    {
      public:
        GeometryPipeline() = default;
        GeometryPipeline( const Context & context );

        GeometryPipeline( const GeometryPipeline & )             = delete;
        GeometryPipeline & operator=( const GeometryPipeline & ) = delete;
        GeometryPipeline( GeometryPipeline && other ) noexcept;
        GeometryPipeline & operator=( GeometryPipeline && ) noexcept;

        ~GeometryPipeline() = default;

        inline void    setRayGen( Module & oModule, std::string name );
        inline void    setMiss( Module & oModule, std::string name );
        SphereModule & add( const Pipeline &      pipeline,
                            std::filesystem::path closestHitModule,
                            std::string           closestHitName );

        void compile() override;
        void updateGeometry();

        inline OptixTraversableHandle getHandle() const;

      private:
        ModuleFunction m_rayGenModule;
        ModuleFunction m_missModule;

        RayGen m_rayGen;
        Miss   m_missGroup;

        std::vector<std::unique_ptr<SphereModule>> m_handles;
        std::vector<OptixInstance>                 m_instances {};

        apo::gpu::DeviceBuffer m_dRayGenRecord;
        apo::gpu::DeviceBuffer m_dMissRecord;
        apo::gpu::DeviceBuffer m_dHitGroupRecord;
        apo::gpu::DeviceBuffer m_dIas;

        OptixTraversableHandle m_handle;
        uint32_t               m_hitGroupRecordNb = 0;
    };
} // namespace apo::optix

#include "optix/geometry_pipeline.inl"

#endif // APO_OPTIX_GEOMETRY_PIPELINE_CUH
