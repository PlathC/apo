#include "optix/geometry.cuh"

namespace apo::optix
{
    OptixTraversableHandle BaseGeometry::getGASHandle() const { return m_gasHandle; }
} // namespace apo::optix
