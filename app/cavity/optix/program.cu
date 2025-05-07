#include <fstream>

#include <optix.h>

#include "optix/pipeline.cuh"
#include "optix/program.cuh"

namespace apo::optix
{
    Module::Module( const Context & context, const std::filesystem::path & modulePath ) :
        m_context( &context ), m_modulePath( std::move( modulePath ) )
    {
    }

    Module::Module( Module && other )
    {
        std::swap( m_context, other.m_context );
        std::swap( m_modulePath, other.m_modulePath );
        std::swap( m_handle, other.m_handle );
        std::swap( m_compiled, other.m_compiled );
    }

    Module & Module::operator=( Module && other )
    {
        std::swap( m_context, other.m_context );
        std::swap( m_modulePath, other.m_modulePath );
        std::swap( m_handle, other.m_handle );
        std::swap( m_compiled, other.m_compiled );

        return *this;
    }

    Module::~Module()
    {
        if ( !m_compiled )
            return;

        optixCheck( optixModuleDestroy( m_handle ) );
    }

    static std::string read( const std::filesystem::path & path )
    {
        std::ifstream file;
        file.open( path, std::ios::in );

        if ( !file.is_open() )
            throw std::runtime_error( "Cannot open file: " + path.string() );

        const uintmax_t size = std::filesystem::file_size( path );
        std::string     result {};
        result.resize( size, '\0' );

        file.read( result.data(), static_cast<std::streamsize>( size ) );
        file.close();

        return result;
    }

    void Module::compile( const Pipeline & pipeline )
    {
        if ( !std::filesystem::exists( m_modulePath ) )
            throw std::runtime_error( fmt::format( "Can't find file: {}", m_modulePath.string() ) );

        const std::string shaderContent = read( m_modulePath );

        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

#ifdef NDEBUG
        moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
        moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif // NDEBUG

        const OptixPipelineCompileOptions pipelineCompileOptions {
            pipeline.m_usesMotionBlur,
            static_cast<unsigned int>( pipeline.m_traversableGraphFlags ),
            static_cast<int>( pipeline.m_numPayloadValues ),
            static_cast<int>( pipeline.m_numAttributeValues ),
            static_cast<unsigned int>( pipeline.m_flags ),
            pipeline.m_variableName.c_str(),
            static_cast<unsigned int>( pipeline.m_primitiveType )
        };

        const std::string src = read( m_modulePath );

        char   log[ 2048 ];
        size_t sizeOfLog = sizeof( log );
        optixCheckLog( optixModuleCreate( m_context->getOptiXContext(),
                                          &moduleCompileOptions,
                                          &pipelineCompileOptions,
                                          shaderContent.data(),
                                          shaderContent.size(),
                                          log,
                                          &sizeOfLog,
                                          &m_handle ) );
        m_compiled = true;
    }

    BuiltinISModule::BuiltinISModule( const Context & context, BuiltinISModuleType type ) :
        Module( context, "" ), m_type( type )
    {
    }

    BuiltinISModule::BuiltinISModule( BuiltinISModule && other ) : Module( std::move( other ) )
    {
        std::swap( m_type, other.m_type );
    }

    BuiltinISModule & BuiltinISModule::operator=( BuiltinISModule && other )
    {
        std::swap( m_type, other.m_type );
        Module::operator=( std::move( other ) );

        return *this;
    }

    BuiltinISModule::~BuiltinISModule() = default;

    void BuiltinISModule::compile( const Pipeline & pipeline )
    {
        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

#ifdef NDEBUG
        moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
        moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif // NDEBUG

        const OptixPipelineCompileOptions pipelineCompileOptions {
            pipeline.m_usesMotionBlur,
            static_cast<unsigned int>( pipeline.m_traversableGraphFlags ),
            static_cast<int>( pipeline.m_numPayloadValues ),
            static_cast<int>( pipeline.m_numAttributeValues ),
            static_cast<unsigned int>( pipeline.m_flags ),
            pipeline.m_variableName.c_str(),
            pipeline.m_primitiveType,
        };

        OptixBuiltinISOptions builtinISOptions = {};
        builtinISOptions.builtinISModuleType   = static_cast<OptixPrimitiveType>( m_type );

        optixCheck( optixBuiltinISModuleGet( //
            m_context->getOptiXContext(),
            &moduleCompileOptions,
            &pipelineCompileOptions,
            &builtinISOptions,
            &m_handle ) );

        m_compiled = true;
    }

    ProgramGroup::ProgramGroup( ProgramGroup && other )
    {
        std::swap( m_handle, other.m_handle );
        std::swap( m_created, other.m_created );
    }

    ProgramGroup & ProgramGroup::operator=( ProgramGroup && other )
    {
        std::swap( m_handle, other.m_handle );
        std::swap( m_created, other.m_created );

        return *this;
    }

    ProgramGroup::~ProgramGroup()
    {
        if ( !m_created )
            return;

        m_created = false;
        optixCheck( optixProgramGroupDestroy( m_handle ) );
    }

    HitGroup::HitGroup( HitGroup && other ) : ProgramGroup( std::move( other ) )
    {
        std::swap( m_isParent, other.m_isParent );
        std::swap( m_isEntryFunctionName, other.m_isEntryFunctionName );
        std::swap( m_chParent, other.m_chParent );
        std::swap( m_chEntryFunctionName, other.m_chEntryFunctionName );
        std::swap( m_ahParent, other.m_ahParent );
        std::swap( m_ahEntryFunctionName, other.m_ahEntryFunctionName );
    }

    HitGroup & HitGroup::operator=( HitGroup && other )
    {
        ProgramGroup::operator=( std::move( other ) );

        std::swap( m_isParent, other.m_isParent );
        std::swap( m_isEntryFunctionName, other.m_isEntryFunctionName );
        std::swap( m_chParent, other.m_chParent );
        std::swap( m_chEntryFunctionName, other.m_chEntryFunctionName );
        std::swap( m_ahParent, other.m_ahParent );
        std::swap( m_ahEntryFunctionName, other.m_ahEntryFunctionName );

        return *this;
    }

    void HitGroup::setIntersection( const Module & parent, std::string entryFunctionName )
    {
        m_isParent            = &parent;
        m_isEntryFunctionName = std::move( entryFunctionName );
    }

    void HitGroup::setClosestHit( const Module & parent, std::string entryFunctionName )
    {
        m_chParent            = &parent;
        m_chEntryFunctionName = std::move( entryFunctionName );
    }

    void HitGroup::setAnyHit( const Module & parent, std::string entryFunctionName )
    {
        m_ahParent            = &parent;
        m_ahEntryFunctionName = std::move( entryFunctionName );
    }

    void HitGroup::create( const Context & context )
    {
        const OptixProgramGroupOptions options = {};

        OptixProgramGroupDesc programGroupDesc = {};
        programGroupDesc.kind                  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        if ( m_isParent )
        {
            programGroupDesc.hitgroup.moduleIS            = m_isParent->getHandle();
            programGroupDesc.hitgroup.entryFunctionNameIS = m_isEntryFunctionName.c_str();
            if ( m_isEntryFunctionName.empty() )
                programGroupDesc.hitgroup.entryFunctionNameIS = nullptr;
        }
        if ( m_chParent )
        {
            programGroupDesc.hitgroup.moduleCH            = m_chParent->getHandle();
            programGroupDesc.hitgroup.entryFunctionNameCH = m_chEntryFunctionName.c_str();
            if ( m_chEntryFunctionName.empty() )
                programGroupDesc.hitgroup.entryFunctionNameCH = nullptr;
        }
        if ( m_ahParent )
        {
            programGroupDesc.hitgroup.moduleAH            = m_ahParent->getHandle();
            programGroupDesc.hitgroup.entryFunctionNameAH = m_ahEntryFunctionName.c_str();
            if ( m_ahEntryFunctionName.empty() )
                programGroupDesc.hitgroup.entryFunctionNameAH = nullptr;
        }

        char        log[ 2048 ];
        std::size_t sizeOfLog = sizeof( log );
        optixCheckLog( optixProgramGroupCreate( context.getOptiXContext(),
                                                &programGroupDesc,
                                                1, // num program groups
                                                &options,
                                                log,
                                                &sizeOfLog,
                                                &m_handle ) );

        m_created = true;
    }

    Miss::Miss( Miss && other ) : ProgramGroup( std::move( other ) )
    {
        std::swap( m_parent, other.m_parent );
        std::swap( m_entryFunctionName, other.m_entryFunctionName );
    }

    Miss & Miss::operator=( Miss && other )
    {
        ProgramGroup::operator=( std::move( other ) );

        std::swap( m_parent, other.m_parent );
        std::swap( m_entryFunctionName, other.m_entryFunctionName );

        return *this;
    }

    void Miss::set( const Module & parent, std::string entryFunctionName )
    {
        m_parent            = &parent;
        m_entryFunctionName = std::move( entryFunctionName );
    }

    void Miss::create( const Context & context )
    {
        const OptixProgramGroupOptions options = {};

        OptixProgramGroupDesc programGroupDesc  = {};
        programGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        programGroupDesc.miss.module            = m_parent ? m_parent->getHandle() : nullptr;
        programGroupDesc.miss.entryFunctionName = m_entryFunctionName.empty() ? nullptr : m_entryFunctionName.c_str();

        char        log[ 2048 ];
        std::size_t sizeOfLog = sizeof( log );
        optixCheckLog( optixProgramGroupCreate( context.getOptiXContext(),
                                                &programGroupDesc,
                                                1, // num program groups
                                                &options,
                                                log,
                                                &sizeOfLog,
                                                &m_handle ) );

        m_created = true;
    }

    RayGen::RayGen( RayGen && other ) : ProgramGroup( std::move( other ) )
    {
        std::swap( m_parent, other.m_parent );
        std::swap( m_entryFunctionName, other.m_entryFunctionName );
    }

    RayGen & RayGen::operator=( RayGen && other )
    {
        ProgramGroup::operator=( std::move( other ) );

        std::swap( m_parent, other.m_parent );
        std::swap( m_entryFunctionName, other.m_entryFunctionName );

        return *this;
    }

    void RayGen::set( const Module & parent, std::string entryFunctionName )
    {
        m_parent            = &parent;
        m_entryFunctionName = std::move( entryFunctionName );
    }

    void RayGen::create( const Context & context )
    {
        const OptixProgramGroupOptions options = {};

        OptixProgramGroupDesc programGroupDesc  = {};
        programGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        programGroupDesc.miss.module            = m_parent->getHandle();
        programGroupDesc.miss.entryFunctionName = m_entryFunctionName.c_str();

        char        log[ 2048 ];
        std::size_t sizeOfLog = sizeof( log );
        optixCheckLog( optixProgramGroupCreate( context.getOptiXContext(),
                                                &programGroupDesc,
                                                1, // num program groups
                                                &options,
                                                log,
                                                &sizeOfLog,
                                                &m_handle ) );

        m_created = true;
    }

} // namespace apo::optix
