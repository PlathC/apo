#ifndef APO_CORE_UTILS_HPP
#define APO_CORE_UTILS_HPP

#include "apo/core/math.hpp"
#include "apo/core/type.hpp"

namespace apo
{
    void joggle( Span<Real> sites, Real range = Real( 1e-4 ) );
}

#endif // APO_CORE_UTILS_HPP
