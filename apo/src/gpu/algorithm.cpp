#include "apo/gpu/algorithm.hpp"

namespace apo::gpu
{
    Algorithm::Algorithm( ConstSpan<Real> sites ) : m_sites( sites ), m_siteNb( uint32_t( m_sites.size / 4 ) )
    {
        for ( std::size_t v = 0; v < m_siteNb; v++ )
        {
            const Real x = m_sites[ v * 4 + 0 ];
            const Real y = m_sites[ v * 4 + 1 ];
            const Real z = m_sites[ v * 4 + 2 ];
            const Real r = m_sites[ v * 4 + 3 ];

            m_minX = apo::min( m_minX, x - r );
            m_minY = apo::min( m_minY, y - r );
            m_minZ = apo::min( m_minZ, z - r );

            m_maxX = apo::max( m_maxX, x + r );
            m_maxY = apo::max( m_maxY, y + r );
            m_maxZ = apo::max( m_maxZ, z + r );

            m_maxRadius = apo::max( m_maxRadius, r );
            m_minRadius = apo::min( m_minRadius, r );
        }
    }
} // namespace apo::gpu
