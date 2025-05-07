#ifndef APO_OPTIX_PROGRAM_CUH
#define APO_OPTIX_PROGRAM_CUH

#include <filesystem>

#include "optix/context.cuh"

namespace apo::optix
{
    class ProgramGroup;
    class HitGroup;
    class Miss;
    class RayGen;
    class Pipeline;

    class Module
    {
      public:
        Module() = default;
        Module( const Context & context, const std::filesystem::path & modulePath );

        Module( const Module & other )             = delete;
        Module & operator=( const Module & other ) = delete;

        Module( Module && other );
        Module & operator=( Module && other );

        virtual ~Module();

        virtual void compile( const Pipeline & pipeline );

        inline OptixModule getHandle() const;

        friend ProgramGroup;
        friend HitGroup;
        friend Miss;
        friend RayGen;

      protected:
        const Context *       m_context;
        std::filesystem::path m_modulePath;
        OptixModule           m_handle;
        bool                  m_compiled = false;
    };

    enum BuiltinISModuleType
    {
        Sphere = OPTIX_PRIMITIVE_TYPE_SPHERE
    };

    class BuiltinISModule : public Module
    {
      public:
        BuiltinISModule() = default;
        BuiltinISModule( const Context & context, BuiltinISModuleType type );

        BuiltinISModule( const BuiltinISModule & other )             = delete;
        BuiltinISModule & operator=( const BuiltinISModule & other ) = delete;

        BuiltinISModule( BuiltinISModule && other );
        BuiltinISModule & operator=( BuiltinISModule && other );

        ~BuiltinISModule();

        void               compile( const Pipeline & pipeline ) override;
        inline OptixModule getHandle() const;

        friend ProgramGroup;
        friend HitGroup;
        friend Miss;
        friend RayGen;

      private:
        BuiltinISModuleType m_type;
    };

    class Pipeline;
    class ProgramGroup
    {
      public:
        ProgramGroup() = default;

        ProgramGroup( const ProgramGroup & )             = delete;
        ProgramGroup & operator=( const ProgramGroup & ) = delete;

        ProgramGroup( ProgramGroup && );
        ProgramGroup & operator=( ProgramGroup && );

        virtual ~ProgramGroup();

        virtual void create( const Context & context ) = 0;

        template<class Type>
        void setSbtRecord( Type & type ) const;

        inline OptixProgramGroup getHandle() const;

        friend Pipeline;

      protected:
        OptixProgramGroup m_handle;
        bool              m_created = false;
    };

    class HitGroup : public ProgramGroup
    {
      public:
        HitGroup() = default;

        HitGroup( HitGroup && );
        HitGroup & operator=( HitGroup && );

        ~HitGroup() override = default;

        void setIntersection( const Module & parent, std::string entryFunctionName );
        void setClosestHit( const Module & parent, std::string entryFunctionName );
        void setAnyHit( const Module & parent, std::string entryFunctionName );

        void create( const Context & context ) override;

      private:
        const Module * m_isParent = nullptr;
        std::string    m_isEntryFunctionName;
        const Module * m_chParent = nullptr;
        std::string    m_chEntryFunctionName;
        const Module * m_ahParent = nullptr;
        std::string    m_ahEntryFunctionName;
    };

    class Miss : public ProgramGroup
    {
      public:
        Miss() = default;

        Miss( Miss && );
        Miss & operator=( Miss && );

        ~Miss() override = default;

        void set( const Module & parent, std::string entryFunctionName );
        void create( const Context & context ) override;

      private:
        const Module * m_parent = nullptr;
        std::string    m_entryFunctionName;
    };

    class RayGen : public ProgramGroup
    {
      public:
        RayGen() = default;

        RayGen( RayGen && );
        RayGen & operator=( RayGen && );

        ~RayGen() override = default;

        void set( const Module & parent, std::string entryFunctionName );
        void create( const Context & context ) override;

      private:
        const Module * m_parent = nullptr;
        std::string    m_entryFunctionName;
    };
} // namespace apo::optix

#include "optix/program.inl"

#endif // APO_OPTIX_PROGRAM_CUH
