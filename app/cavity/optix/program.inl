#include "optix/program.cuh"

namespace apo::optix
{
    inline OptixModule Module::getHandle() const { return m_handle; }

    template<class Type>
    void ProgramGroup::setSbtRecord( Type & type ) const
    {
        optixCheck( optixSbtRecordPackHeader( m_handle, &type ) );
    }

    inline OptixProgramGroup ProgramGroup::getHandle() const { return m_handle; }
} // namespace apo::optix
