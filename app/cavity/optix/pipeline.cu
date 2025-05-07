#include "optix/pipeline.cuh"

//
#include <cassert>

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

namespace apo::optix
{
    Pipeline::Pipeline( const Context & context ) :
        m_context( &context ), m_traversableGraphFlags( OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY ),
        m_flags( OPTIX_EXCEPTION_FLAG_NONE )
    {
    }

    Pipeline::Pipeline( Pipeline && other ) noexcept
    {
        std::swap( m_context, other.m_context );
        std::swap( m_handle, other.m_handle );
        std::swap( m_compiled, other.m_compiled );
        std::swap( m_programGroups, other.m_programGroups );

        std::swap( m_usesMotionBlur, other.m_usesMotionBlur );
        std::swap( m_traversableGraphFlags, other.m_traversableGraphFlags );
        std::swap( m_numPayloadValues, other.m_numPayloadValues );
        std::swap( m_numAttributeValues, other.m_numAttributeValues );
        std::swap( m_flags, other.m_flags );
        std::swap( m_variableName, other.m_variableName );
        std::swap( m_primitiveType, other.m_primitiveType );

        std::swap( m_maxTraceDepth, other.m_maxTraceDepth );
        std::swap( m_maxTraversalDepth, other.m_maxTraversalDepth );
    }

    Pipeline & Pipeline::operator=( Pipeline && other ) noexcept
    {
        std::swap( m_context, other.m_context );
        std::swap( m_handle, other.m_handle );
        std::swap( m_compiled, other.m_compiled );
        std::swap( m_programGroups, other.m_programGroups );

        std::swap( m_usesMotionBlur, other.m_usesMotionBlur );
        std::swap( m_traversableGraphFlags, other.m_traversableGraphFlags );
        std::swap( m_numPayloadValues, other.m_numPayloadValues );
        std::swap( m_numAttributeValues, other.m_numAttributeValues );
        std::swap( m_flags, other.m_flags );
        std::swap( m_variableName, other.m_variableName );
        std::swap( m_primitiveType, other.m_primitiveType );

        std::swap( m_maxTraceDepth, other.m_maxTraceDepth );
        std::swap( m_maxTraversalDepth, other.m_maxTraversalDepth );

        return *this;
    }

    Pipeline::~Pipeline()
    {
        if ( !m_compiled )
            return;

        optixPipelineDestroy( m_handle );
    }

    void Pipeline::compile()
    {
        if ( m_compiled )
            optixPipelineDestroy( m_handle );

        const OptixPipelineLinkOptions pipelineLinkOptions {
            m_maxTraceDepth, // maxTraceDepth
        };

        const OptixPipelineCompileOptions pipelineCompileOptions { m_usesMotionBlur,
                                                                   m_traversableGraphFlags,
                                                                   static_cast<int>( m_numPayloadValues ),
                                                                   static_cast<int>( m_numAttributeValues ),
                                                                   m_flags,
                                                                   m_variableName.c_str(),
                                                                   static_cast<unsigned int>( m_primitiveType ) };

        char   log[ 2048 ];
        size_t sizeOfLog = sizeof( log );
        optixCheckLog( optixPipelineCreate( m_context->getOptiXContext(),
                                            &pipelineCompileOptions,
                                            &pipelineLinkOptions,
                                            m_programGroups.data(),
                                            static_cast<uint32_t>( m_programGroups.size() ),
                                            log,
                                            &sizeOfLog,
                                            &m_handle ) );

        OptixStackSizes stackSizes {};
        for ( const auto & programGroup : m_programGroups )
            optixCheck( optixUtilAccumulateStackSizes( programGroup, &stackSizes, m_handle ) );

        uint32_t directCallableStackSizeFromTraversal;
        uint32_t directCallableStackSizeFromState;
        uint32_t continuationStackSize;
        optixCheck( optixUtilComputeStackSizes( &stackSizes,
                                                m_maxTraceDepth,
                                                0, // maxCCDepth
                                                0, // maxDCDepth
                                                &directCallableStackSizeFromTraversal,
                                                &directCallableStackSizeFromState,
                                                &continuationStackSize ) );
        optixCheck( optixPipelineSetStackSize( m_handle,
                                               directCallableStackSizeFromTraversal,
                                               directCallableStackSizeFromState,
                                               continuationStackSize,
                                               m_maxTraversalDepth ) );

        m_compiled = true;
    }

    void Pipeline::launch( const uint8_t *                 pipelineParams,
                           std::size_t                     pipelineParamsSize,
                           const OptixShaderBindingTable & sbt,
                           unsigned int                    width,
                           unsigned int                    height,
                           unsigned int                    depth )
    {
        assert( m_compiled );
        optixCheck( optixLaunch( m_handle,
                                 m_context->getStream(),
                                 reinterpret_cast<CUdeviceptr>( pipelineParams ),
                                 pipelineParamsSize,
                                 &sbt,
                                 width,
                                 height,
                                 depth ) );
    }

} // namespace apo::optix
