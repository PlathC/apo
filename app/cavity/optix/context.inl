#include "optix/context.cuh"

namespace apo::optix
{
    inline CUstream           Context::getStream() const { return m_stream; }
    inline OptixDeviceContext Context::getOptiXContext() const { return m_optiXContext; }
} // namespace apo::optix