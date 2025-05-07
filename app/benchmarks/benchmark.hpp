#ifndef APO_APP_STATISTICS_HPP
#define APO_APP_STATISTICS_HPP

#include <apo/core/math.hpp>
#include <apo/core/type.hpp>

namespace apo
{
    std::vector<double> benchmark( uint32_t warmupNb, uint32_t sampleNb, apo::ConstSpan<apo::Real> sites );
}

#endif // APO_APP_STATISTICS_HPP