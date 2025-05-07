#ifndef APO_PRECISION_PRECISION_HPP
#define APO_PRECISION_PRECISION_HPP

#include <vector>

#include <apo/gpu/algorithm.hpp>

namespace apo::precision
{
    struct Quadrisector
    {
        uint32_t i, j, k, l;
        double   x, y, z;
    };

    std::size_t getBadQuadrisectorNb( const apo::ConstSpan<Real>        sites,
                                      const std::vector<Quadrisector> & quadrisectors );

    struct Trisector
    {
        uint32_t i, j, k;
        double   x, y, z;
    };
    std::size_t getBadTrisectorNb( const apo::ConstSpan<Real> sites, const std::vector<Trisector> & closedTrisectors );
} // namespace apo::precision

#endif // APO_PRECISION_PRECISION_HPP