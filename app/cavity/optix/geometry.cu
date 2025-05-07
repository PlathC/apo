#include "optix/geometry.cuh"

namespace apo::optix
{
    BaseGeometry::BaseGeometry( const Context & context ) : m_context( &context ) {}
} // namespace apo::optix
