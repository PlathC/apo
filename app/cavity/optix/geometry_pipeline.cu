#include "optix/geometry_pipeline.cuh"
#include "optix/sphere_module.cuh"

namespace apo::optix
{
    GeometryPipeline::GeometryPipeline( const Context & context ) : Pipeline( context )
    {
        setUsesMotionBlur( false );
        setTraversableGraphFlags( OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING );
        setNumPayloadValues( 2 );
        setNumAttributeValues( 8 );

#ifndef NDEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should
        // only be done during development.
        setExceptionFlags( OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW );
#else
        setExceptionFlags( OPTIX_EXCEPTION_FLAG_NONE );
#endif
        setMaxTraversalDepth( 2 );
        setMaxTraceDepth( 4 );
        setPipelineLaunchParamsVariableName( "parameters" );
    }

    GeometryPipeline::GeometryPipeline( GeometryPipeline && other ) noexcept
    {
        std::swap( m_rayGenModule, other.m_rayGenModule );
        std::swap( m_rayGen, other.m_rayGen );
        std::swap( m_missGroup, other.m_missGroup );
        std::swap( m_handles, other.m_handles );
        std::swap( m_dRayGenRecord, other.m_dRayGenRecord );
        std::swap( m_dMissRecord, other.m_dMissRecord );
        std::swap( m_dHitGroupRecord, other.m_dHitGroupRecord );
        std::swap( m_handle, other.m_handle );
        std::swap( m_hitGroupRecordNb, other.m_hitGroupRecordNb );
    }

    GeometryPipeline & GeometryPipeline::operator=( GeometryPipeline && other ) noexcept
    {
        std::swap( m_rayGenModule, other.m_rayGenModule );
        std::swap( m_rayGen, other.m_rayGen );
        std::swap( m_missGroup, other.m_missGroup );
        std::swap( m_handles, other.m_handles );
        std::swap( m_dRayGenRecord, other.m_dRayGenRecord );
        std::swap( m_dMissRecord, other.m_dMissRecord );
        std::swap( m_dHitGroupRecord, other.m_dHitGroupRecord );
        std::swap( m_handle, other.m_handle );
        std::swap( m_hitGroupRecordNb, other.m_hitGroupRecordNb );

        return *this;
    }

    SphereModule & GeometryPipeline::add( const Pipeline &      pipeline,
                                          std::filesystem::path closestHitModule,
                                          std::string           closestHitName )
    {
        auto           derivedHandler = std::make_unique<SphereModule>( pipeline, closestHitModule, closestHitName );
        SphereModule * ptr            = derivedHandler.get();
        m_handles.emplace_back( std::move( derivedHandler ) );
        return *ptr;
    }

    void GeometryPipeline::compile()
    {
        m_rayGen = {};
        m_rayGen.set( *m_rayGenModule.first, m_rayGenModule.second );
        m_rayGen.create( *m_context );
        addProgramGroup( m_rayGen );

        m_missGroup = {};
        m_missGroup.set( *m_missModule.first, m_missModule.second );
        m_missGroup.create( *m_context );
        addProgramGroup( m_missGroup );

        m_hitGroupRecordNb = 0;
        for ( const auto & handle : m_handles )
            m_hitGroupRecordNb += handle->getHitGroupRecordNb();

        std::vector<GeometryHitGroupRecord> hitGroupRecords {};
        hitGroupRecords.reserve( m_hitGroupRecordNb );
        m_dHitGroupRecord = apo::gpu::DeviceBuffer::Typed<GeometryHitGroupRecord>( m_hitGroupRecordNb );

        // Process every HitGroups modules
        uint32_t sbtOffset      = 0;
        uint32_t instanceOffset = 0;
        m_instances.clear();
        m_instances.reserve( m_handles.size() );
        for ( const auto & handle : m_handles )
        {
            const auto & hitGroups = handle->getHitGroups();
            auto         records   = handle->getRecords();

            if ( records.empty() )
                continue;

            for ( const auto & hitGroup : hitGroups )
                addProgramGroup( *hitGroup );

            hitGroupRecords.insert( hitGroupRecords.end(),
                                    std::make_move_iterator( records.begin() ),
                                    std::make_move_iterator( records.end() ) );

            auto currentInstances = handle->getInstances( instanceOffset, sbtOffset );
            if ( !currentInstances.empty() )
            {
                m_instances.insert( m_instances.end(),
                                    std::make_move_iterator( currentInstances.begin() ),
                                    std::make_move_iterator( currentInstances.end() ) );
            }
        }

        apo::gpu::cudaCheck( cudaMemcpy( m_dHitGroupRecord.get<uint8_t>(),
                                         hitGroupRecords.data(),
                                         sizeof( GeometryHitGroupRecord ) * m_hitGroupRecordNb,
                                         cudaMemcpyHostToDevice ) );

        Pipeline::compile();
    }

    void GeometryPipeline::updateGeometry()
    {
        struct Dummy
        {
        };

        m_dRayGenRecord            = apo::gpu::DeviceBuffer::Typed<optix::Record<Dummy>>( 1u );
        optix::Record<Dummy> rgSbt = {};
        {
            m_rayGen.setSbtRecord( rgSbt );
            apo::gpu::cudaCheck( cudaMemcpy(
                m_dRayGenRecord.get<uint8_t>(), &rgSbt, sizeof( optix::Record<Dummy> ), cudaMemcpyHostToDevice ) );
        }

        m_dMissRecord = apo::gpu::DeviceBuffer::Typed<optix::Record<Dummy>>( 1 );
        {
            optix::Record<Dummy> missRecord {};
            m_missGroup.setSbtRecord( missRecord );

            apo::gpu::cudaCheck( cudaMemcpy(
                m_dMissRecord.get<uint8_t>(), &missRecord, sizeof( optix::Record<Dummy> ), cudaMemcpyHostToDevice ) );
        }

        m_sbt.exceptionRecord              = 0;
        m_sbt.callablesRecordBase          = 0;
        m_sbt.callablesRecordCount         = 0;
        m_sbt.callablesRecordStrideInBytes = 0;

        m_sbt.raygenRecord            = reinterpret_cast<CUdeviceptr>( m_dRayGenRecord.get() );
        m_sbt.missRecordBase          = reinterpret_cast<CUdeviceptr>( m_dMissRecord.get() );
        m_sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof( optix::Record<Dummy> ) );
        m_sbt.missRecordCount         = 1;

        m_sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>( m_dHitGroupRecord.get() );
        m_sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( sizeof( GeometryHitGroupRecord ) );
        m_sbt.hitgroupRecordCount         = m_hitGroupRecordNb;

        const size_t instancesSizeInBytes = sizeof( OptixInstance ) * m_instances.size();
        auto         dInstances
            = apo::gpu::DeviceBuffer( instancesSizeInBytes + instancesSizeInBytes % OPTIX_INSTANCE_BYTE_ALIGNMENT );

        cudaMemcpy( dInstances.get(), m_instances.data(), instancesSizeInBytes, cudaMemcpyHostToDevice );

        OptixBuildInput instanceInput            = {};
        instanceInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instanceInput.instanceArray.instances    = reinterpret_cast<CUdeviceptr>( dInstances.get() );
        instanceInput.instanceArray.numInstances = static_cast<uint32_t>( m_instances.size() );

        OptixAccelBuildOptions accelBuildOptions = {};
        accelBuildOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
        accelBuildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes accelBufferSizes;
        optixCheck( optixAccelComputeMemoryUsage(
            m_context->getOptiXContext(), &accelBuildOptions, &instanceInput, 1, &accelBufferSizes ) );

        m_dIas = apo::gpu::DeviceBuffer( accelBufferSizes.outputSizeInBytes
                                         + accelBufferSizes.outputSizeInBytes % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT );

        auto dTmp = apo::gpu::DeviceBuffer( accelBufferSizes.tempSizeInBytes
                                            + accelBufferSizes.tempSizeInBytes % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT );

        optixCheck( optixAccelBuild( m_context->getOptiXContext(),
                                     m_context->getStream(),
                                     &accelBuildOptions,
                                     &instanceInput,
                                     1,
                                     reinterpret_cast<CUdeviceptr>( dTmp.get() ),
                                     accelBufferSizes.tempSizeInBytes,
                                     reinterpret_cast<CUdeviceptr>( m_dIas.get() ),
                                     accelBufferSizes.outputSizeInBytes,
                                     &m_handle,
                                     nullptr,
                                     0 ) );
    }

} // namespace apo::optix
