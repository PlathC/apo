#ifndef APO_CORE_MATH_HPP
#define APO_CORE_MATH_HPP

#include <cmath>
#include <cstdint>

#include "apo/gpu/setup.hpp"

#ifndef APO_REAL_SIZE
    #define APO_REAL_SIZE 8
#endif // APO_REAL_SIZE

namespace apo
{
    // Reference:
    // https://stackoverflow.com/questions/14637356/static-assert-fails-compilation-even-though-template-function-is-called-nowhere
    template<typename...>
    inline constexpr bool AlwaysFalse = false;

    static_assert( sizeof( uint8_t ) == 1 );
    static_assert( sizeof( uint16_t ) == 2 );
    static_assert( sizeof( uint32_t ) == 4 );
    static_assert( sizeof( uint64_t ) == 8 );
    static_assert( sizeof( float ) == 4 );
    static_assert( sizeof( double ) == 8 );

#if APO_REAL_SIZE == 8
    using Real = double;
#else
    using Real = float;
#endif // APO_REAL_SIZE

    constexpr Real  Pi       = 3.1415926535;
    constexpr Real  TwoPi    = 6.2831853071;
    constexpr Real  Epsilon  = 1e-6;
    constexpr Real  Max      = std::numeric_limits<Real>::max();
    constexpr Real  Lowest   = std::numeric_limits<Real>::lowest();
    constexpr float MaxFloat = std::numeric_limits<float>::max();

    template<class Type>
    APO_HOST APO_DEVICE Type sqrt( Type v );
    template<class Type>
    APO_HOST APO_DEVICE constexpr Type pow2( Type v );
    template<class Type>
    APO_HOST APO_DEVICE constexpr Type min( Type a, Type b );
    template<class Type>
    APO_HOST APO_DEVICE constexpr Type max( Type a, Type b );
    template<class Type>
    APO_HOST APO_DEVICE constexpr Type abs( Type a );
    template<class Type>
    APO_HOST APO_DEVICE constexpr Type clamp( Type x, Type minVal, Type maxVal );
    template<class Type>
    APO_HOST APO_DEVICE constexpr Type floor( Type x );
    template<class Type>
    APO_HOST APO_DEVICE constexpr Type ceil( Type x );

    template<class Type>
    APO_HOST APO_DEVICE constexpr Type acos( Type x );
    template<class Type>
    APO_HOST APO_DEVICE constexpr Type asinh( Type x );
    template<class Type>
    APO_HOST APO_DEVICE constexpr Type atan2( Type a, Type b );

    template<class Type>
    APO_HOST APO_DEVICE inline bool isClose( Type a, Type reference, Type = Type( 1e-5 ), Type rtol = Type( 1e-8 ) );
    template<class Type>
    APO_HOST APO_DEVICE constexpr bool isCloseToZero( Type value, Type eps = Type( 1e-8 ) );
    template<class Type>
    APO_HOST APO_DEVICE constexpr bool lessThan( Type a, Type b, Type eps = Type( 1e-8 ) );
    template<class Type>
    APO_HOST APO_DEVICE constexpr bool greaterThan( Type a, Type b, Type eps = Type( 1e-8 ) );
    template<class Type>
    APO_HOST APO_DEVICE constexpr Type sign( Type a );

    template<class Type>
    APO_HOST APO_DEVICE uint32_t popcount( Type a );
    template<class Type>
    APO_HOST APO_DEVICE uint32_t ffs( Type a );

} // namespace apo

#include "apo/core/math.inl"

#endif // APO_CORE_MATH_HPP
