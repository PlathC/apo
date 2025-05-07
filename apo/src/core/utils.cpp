#include "apo/core/utils.hpp"

namespace apo
{
    // Reference: https://www.shadertoy.com/view/XlGcRh
    apo::Real pcg( uint32_t v )
    {
        uint32_t state = v * 747796405u + 2891336453u;
        uint32_t word  = ( ( state >> ( ( state >> 28u ) + 4u ) ) ^ state ) * 277803737u;
        return static_cast<apo::Real>( std::ldexp( ( word >> 22u ) ^ word, -32 ) );
    }

    // Reference:
    // https://github.com/qhull/qhull/blob/c2ef2209c28dc61ccfd22514971236587e820121/src/libqhull/geom2.c#L1036
    void joggle( Span<apo::Real> sites, apo::Real range )
    {
        const std::size_t count = sites.size / 4;
        const apo::Real   randA = apo::Real( 2 ) * range;
        const apo::Real   randB = -range;
        for ( uint32_t i = 0; i < count; i++ )
        {
            for ( uint8_t j = 0; j < 4; j++ )
                sites[ i * 4 + j ] += pcg( i * 4 + j ) * randA + randB;
        }
    }
} // namespace apo