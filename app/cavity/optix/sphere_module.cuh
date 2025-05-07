#ifndef APO_OPTIX_SPHERE_MODULE_HPP
#define APO_OPTIX_SPHERE_MODULE_HPP

#include "apo/gpu/memory.cuh"
#include "optix/geometry.cuh"
#include "optix/geometry_pipeline.cuh"
#include "optix/program.cuh"
#include "shaders/data.cuh"

namespace apo::optix
{
    class SphereGeometry : public BaseGeometry
    {
      public:
        SphereGeometry( const Context & optixContext, ConstSpan<float> spheres );

        void build() override;

        uint32_t getGeometryNb() const override;

        std::vector<GeometryHitGroup> getGeometryData() const;

      private:
        ConstSpan<float> m_spheres;

        gpu::DeviceBuffer m_dGasOutputBuffer;
        gpu::DeviceBuffer m_data;
    };

    class SphereModule
    {
      public:
        SphereModule( const Pipeline & pipeline, std::filesystem::path closestHitModule, std::string closestHitName );

        std::vector<GeometryHitGroupRecord> getRecords();

        std::vector<OptixInstance> getInstances( uint32_t & instanceOffset, uint32_t & sbtOffset );

        uint32_t getHitGroupRecordNb() const;

        std::vector<const HitGroup *> getHitGroups() const;

        void add( SphereGeometry & geometry );

      private:
        const Context * m_context;

        std::vector<SphereGeometry *> m_geometries;

        Module          m_sphereModule;
        BuiltinISModule m_sphereISModule;
        HitGroup        m_sphereHitGroup;
    };
} // namespace apo::optix

#endif // APO_OPTIX_SPHERE_MODULE_CUH
