#include "optix/pipeline.cuh"
#include "optix/sphere_module.cuh"

namespace apo::optix
{
    SphereGeometry::SphereGeometry( const Context & optixContext, ConstSpan<float> spheres ) :
        BaseGeometry( optixContext ), m_spheres( spheres )
    {
    }

    void SphereGeometry::build()
    {
        const uint32_t primitiveNb = getGeometryNb();

        auto dSbtIndex = apo::gpu::DeviceBuffer::Typed<uint32_t>( primitiveNb );
        apo::gpu::cudaCheck( cudaMemset( dSbtIndex.get(), 0, dSbtIndex.size() ) );

        if ( m_data.size<float4>() < primitiveNb )
            m_data = apo::gpu::DeviceBuffer::Typed<float4>( primitiveNb );

        apo::gpu::cudaCheck(
            cudaMemcpy( m_data.get(), m_spheres.ptr, sizeof( float ) * m_spheres.size, cudaMemcpyHostToDevice ) );

        OptixBuildInput aabbInput = {};
        aabbInput.type            = OPTIX_BUILD_INPUT_TYPE_SPHERES;

        aabbInput.sphereArray.numVertices          = primitiveNb;
        aabbInput.sphereArray.primitiveIndexOffset = 0;

        const CUdeviceptr spherePtr               = reinterpret_cast<CUdeviceptr>( m_data.get() );
        aabbInput.sphereArray.vertexBuffers       = &spherePtr;
        aabbInput.sphereArray.vertexStrideInBytes = sizeof( float4 );

        const CUdeviceptr radiusPtr               = reinterpret_cast<CUdeviceptr>( m_data.get<float>() + 3 );
        aabbInput.sphereArray.radiusBuffers       = &radiusPtr;
        aabbInput.sphereArray.radiusStrideInBytes = sizeof( float4 );

        const uint32_t aabbInputFlags                     = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT; // One per SBt
        aabbInput.sphereArray.numSbtRecords               = 1;
        aabbInput.sphereArray.flags                       = &aabbInputFlags;
        aabbInput.sphereArray.sbtIndexOffsetBuffer        = reinterpret_cast<CUdeviceptr>( dSbtIndex.get() );
        aabbInput.sphereArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
        aabbInput.sphereArray.sbtIndexOffsetStrideInBytes = 0;

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gasBufferSizes;

        optixCheck( optixAccelComputeMemoryUsage(
            m_context->getOptiXContext(), &accelOptions, &aabbInput, 1, &gasBufferSizes ) );

        auto dTempBufferGas = apo::gpu::DeviceBuffer::Typed<uint8_t>( gasBufferSizes.tempSizeInBytes );

        // non-compacted output and size of compacted GAS
        const std::size_t compactedSizeOffset = ( ( gasBufferSizes.outputSizeInBytes + 8ull - 1ull ) / 8ull ) * 8ull;
        auto dBufferTempOutputGasAndCompactedSize = apo::gpu::DeviceBuffer::Typed<uint8_t>( compactedSizeOffset + 8 );

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result
            = reinterpret_cast<CUdeviceptr>( dBufferTempOutputGasAndCompactedSize.get() + compactedSizeOffset );

        optixCheck( optixAccelBuild( m_context->getOptiXContext(),
                                     m_context->getStream(),
                                     &accelOptions,
                                     &aabbInput,
                                     1,
                                     reinterpret_cast<CUdeviceptr>( dTempBufferGas.get() ),
                                     gasBufferSizes.tempSizeInBytes,
                                     reinterpret_cast<CUdeviceptr>( dBufferTempOutputGasAndCompactedSize.get() ),
                                     gasBufferSizes.outputSizeInBytes,
                                     &m_gasHandle,
                                     &emitProperty,
                                     1 ) );

        std::size_t compactedGasSize;
        apo::gpu::cudaCheck( cudaMemcpy(
            &compactedGasSize, (void *)emitProperty.result, sizeof( std::size_t ), cudaMemcpyDeviceToHost ) );
        if ( compactedGasSize < gasBufferSizes.outputSizeInBytes )
        {
            m_dGasOutputBuffer = apo::gpu::DeviceBuffer::Typed<uint8_t>( compactedGasSize );

            // use handle as input and output
            optixCheck( optixAccelCompact( m_context->getOptiXContext(),
                                           m_context->getStream(),
                                           m_gasHandle,
                                           reinterpret_cast<CUdeviceptr>( m_dGasOutputBuffer.get() ),
                                           compactedGasSize,
                                           &m_gasHandle ) );
        }
        else
        {
            m_dGasOutputBuffer = std::move( dBufferTempOutputGasAndCompactedSize );
        }
    }

    uint32_t SphereGeometry::getGeometryNb() const { return static_cast<uint32_t>( m_spheres.size / 4 ); }

    std::vector<GeometryHitGroup> SphereGeometry::getGeometryData() const
    {
        GeometryHitGroup hitGroup;

        // 1 SBT per geometry type
        return { hitGroup };
    }

    SphereModule::SphereModule( const Pipeline &      pipeline,
                                std::filesystem::path closestHitModule,
                                std::string           closestHitName ) :
        m_context( &pipeline.getContext() )
    {
        m_sphereModule = Module( *m_context, closestHitModule );
        m_sphereModule.compile( pipeline );

        m_sphereISModule = BuiltinISModule( *m_context, BuiltinISModuleType::Sphere );
        m_sphereISModule.compile( pipeline );

        m_sphereHitGroup.setIntersection( m_sphereISModule, "" );
        m_sphereHitGroup.setClosestHit( m_sphereModule, std::move( closestHitName ) );
        m_sphereHitGroup.create( *m_context );
    }

    std::vector<GeometryHitGroupRecord> SphereModule::getRecords()
    {
        std::vector<GeometryHitGroupRecord> geometriesHitGroup {};
        for ( auto geometry : m_geometries )
        {
            geometry->build();

            std::vector<GeometryHitGroup> geometryData = geometry->getGeometryData();

            geometriesHitGroup.emplace_back();
            auto & sphere = geometriesHitGroup.back();
            m_sphereHitGroup.setSbtRecord( sphere );
            sphere.data = geometryData[ 0 ];
        }

        return geometriesHitGroup;
    }

    std::vector<OptixInstance> SphereModule::getInstances( uint32_t & instanceOffset, uint32_t & sbtOffset )
    {
        std::vector<OptixInstance> instances {};
        for ( auto geometry : m_geometries )
        {
            instances.emplace_back();
            OptixInstance & instance   = instances.back();
            instance.instanceId        = instanceOffset;
            instance.sbtOffset         = sbtOffset;
            instance.visibilityMask    = 255;
            instance.traversableHandle = geometry->getGASHandle();
            instance.flags             = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;

            constexpr float init[ 12 ] = {
                1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f,
            };
            std::memcpy( instance.transform, init, sizeof( float ) * 12 );

            instanceOffset++;
            sbtOffset++;
        }

        return instances;
    }

    uint32_t SphereModule::getHitGroupRecordNb() const { return m_geometries.size(); }

    std::vector<const HitGroup *> SphereModule::getHitGroups() const
    {
        std::vector<const HitGroup *> hitGroups { 1 };
        hitGroups[ 0 ] = &m_sphereHitGroup;

        return hitGroups;
    }

    void SphereModule::add( SphereGeometry & geometry ) { m_geometries.emplace_back( &geometry ); }
} // namespace apo::optix
