#include "optix/pipeline.cuh"

namespace apo::optix
{
    void Pipeline::setUsesMotionBlur( bool value ) { m_usesMotionBlur = value; }
    void Pipeline::setTraversableGraphFlags( const uint32_t value ) { m_traversableGraphFlags = value; }
    void Pipeline::setNumPayloadValues( const uint32_t numPayloadValues ) { m_numPayloadValues = numPayloadValues; }
    void Pipeline::setNumAttributeValues( const uint32_t numAttributeValues )
    {
        m_numAttributeValues = numAttributeValues;
    }
    void Pipeline::setExceptionFlags( const uint32_t flags ) { m_flags = flags; }
    void Pipeline::setPipelineLaunchParamsVariableName( std::string variableName )
    {
        m_variableName = std::move( variableName );
    }
    void Pipeline::setPrimitiveType( const uint32_t flags ) { m_primitiveType = flags; }

    void Pipeline::setMaxTraceDepth( const uint32_t depth ) { m_maxTraceDepth = depth; }

    void Pipeline::setMaxTraversalDepth( uint32_t depth ) { m_maxTraversalDepth = depth; }

    void Pipeline::addProgramGroup( const ProgramGroup & programGroup )
    {
        m_programGroups.emplace_back( programGroup.getHandle() );
    }

    inline OptixShaderBindingTable Pipeline::getBindingTable() const { return m_sbt; }
    inline const Context &         Pipeline::getContext() const { return *m_context; }
} // namespace apo::optix
