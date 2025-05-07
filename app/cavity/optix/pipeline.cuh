#ifndef APO_OPTIX_PIPELINE_CUH
#define APO_OPTIX_PIPELINE_CUH

#include "optix/program.cuh"

namespace apo::optix
{
    class Pipeline
    {
      public:
        Pipeline() = default;
        Pipeline( const Context & context );

        Pipeline( const Pipeline & )             = delete;
        Pipeline & operator=( const Pipeline & ) = delete;

        Pipeline( Pipeline && ) noexcept;
        Pipeline & operator=( Pipeline && ) noexcept;

        virtual ~Pipeline();

        inline void setUsesMotionBlur( bool value );
        inline void setTraversableGraphFlags( uint32_t value );
        inline void setNumPayloadValues( uint32_t numPayloadValues );
        inline void setNumAttributeValues( uint32_t numAttributeValues );
        inline void setExceptionFlags( uint32_t flags );
        inline void setPipelineLaunchParamsVariableName( std::string variableName );
        inline void setPrimitiveType( uint32_t flags );
        inline void setMaxTraceDepth( uint32_t depth );
        inline void setMaxTraversalDepth( uint32_t depth );
        inline void addProgramGroup( const ProgramGroup & p_programGroup );

        inline OptixShaderBindingTable getBindingTable() const;
        inline const Context &         getContext() const;

        virtual void compile();
        virtual void launch( const uint8_t *                 pipelineParams,
                             std::size_t                     pipelineParamsSize,
                             const OptixShaderBindingTable & sbt,
                             unsigned int                    width,
                             unsigned int                    height,
                             unsigned int                    depth );

        friend Module;
        friend BuiltinISModule;

      protected:
        const Context *         m_context;
        OptixShaderBindingTable m_sbt;

      private:
        OptixPipeline m_handle;

        std::vector<OptixProgramGroup> m_programGroups {};

        bool        m_usesMotionBlur = false;
        uint32_t    m_traversableGraphFlags;
        uint32_t    m_numPayloadValues;
        uint32_t    m_numAttributeValues;
        uint32_t    m_flags;
        std::string m_variableName;
        uint32_t    m_primitiveType;

        uint32_t m_maxTraceDepth     = 12;
        uint32_t m_maxTraversalDepth = 1;
        bool     m_compiled          = false;
    };
} // namespace apo::optix

#include "optix/pipeline.inl"

#endif // APO_OPTIX_PIPELINE_CUH
