#include "apo/core/math.hpp"

namespace apo
{
    template<class Type>
    APO_HOST APO_DEVICE

        Type
        sqrt( Type v )
    {
#ifndef __CUDACC__
        return std::sqrt( v );
#else
#if AAPO_REAL_SIZE == 8
        return ::sqrt( v );
#else
        return ::sqrtf( v );
#endif // AAPO_REAL_SIZE
#endif // __CUDACC__
    }

    template<class Type>
    APO_HOST APO_DEVICE constexpr Type pow2( Type v )
    {
        return v * v;
    }

    template<class Type>
    APO_HOST APO_DEVICE constexpr Type min( Type a, Type b )
    {
#ifndef __CUDACC__
        return std::min( a, b );
#else
        return ::min( a, b );
#endif // __CUDACC__
    }

    template<class Type>
    APO_HOST APO_DEVICE constexpr Type max( Type a, Type b )
    {
#ifndef __CUDACC__
        return std::max( a, b );
#else
        return ::max( a, b );
#endif // __CUDACC__
    }

    template<class Type>
    APO_HOST APO_DEVICE constexpr Type abs( Type a )
    {
#ifndef __CUDACC__
        return std::abs( a );
#else
#if AAPO_REAL_SIZE == 8
        return ::abs( a );
#else
        return ::fabs( a );
#endif // AAPO_REAL_SIZE
#endif // __CUDACC__
    }

    template<class Type>
    APO_HOST APO_DEVICE constexpr Type clamp( Type x, Type minVal, Type maxVal )
    {
        return apo::min( apo::max( x, minVal ), maxVal );
    }

    template<class Type>
    APO_HOST APO_DEVICE constexpr Type floor( Type x )
    {
#ifndef __CUDACC__
        return std::floor( x );
#else
        return ::floor( x );
#endif // __CUDACC__
    }

    template<class Type>
    APO_HOST APO_DEVICE constexpr Type ceil( Type x )
    {
#ifndef __CUDACC__
        return std::ceil( x );
#else
        return ::ceil( x );
#endif // __CUDACC__
    }

    template<class Type>
    APO_HOST APO_DEVICE constexpr Type acos( Type x )
    {
#ifndef __CUDACC__
        return std::acos( x );
#else
        return ::acos( x );
#endif // __CUDACC__
    }

    template<class Type>
    APO_HOST APO_DEVICE constexpr Type asinh( Type x )
    {
#ifndef __CUDACC__
        return std::asinh( x );
#else
        return ::asinh( x );
#endif // __CUDACC__
    }

    template<class Type>
    APO_HOST APO_DEVICE constexpr Type atan2( Type a, Type b )
    {
#ifndef __CUDACC__
        return std::atan2( a, b );
#else
        return ::atan2( a, b );
#endif // __CUDACC__
    }

    template<class Type>
    APO_HOST APO_DEVICE

        inline bool
        isClose( Type a, Type reference, Type atol, Type rtol )
    {
        // See: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
        // See: https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/numeric.py#L2330
        return apo::abs( a - reference ) <= ( atol + rtol * apo::abs( reference ) );
    }

    template<class Type>
    APO_HOST APO_DEVICE

        constexpr bool
        isCloseToZero( Type value, Type eps )
    {
        return abs( value ) < eps;
    }

    template<class Type>
    APO_HOST APO_DEVICE

        constexpr bool
        lessThan( Type a, Type b, Type eps )
    {
        return ( a - b ) < eps;
    }

    template<class Type>
    APO_HOST APO_DEVICE

        constexpr bool
        greaterThan( Type a, Type b, Type eps )
    {
        return ( a - b ) > -eps;
    }

    template<class Type>
    APO_HOST APO_DEVICE constexpr Type sign( Type a )
    {
        return lessThan( a, Type( 0 ) ) ? Type( -1 ) : Type( 1 );
    }

    template<class Type>
    APO_HOST APO_DEVICE

        uint32_t
        popcount( uint32_t a )
    {
        static_assert( AlwaysFalse<Type>, "Popcount is not implemented for this type." );
        return 0;
    }

    template<>
    inline APO_HOST APO_DEVICE

        uint32_t
        popcount( uint32_t a )
    {
#if defined( __GNUC__ ) || defined( __clang__ )
        return __builtin_popcount( a );
#elif defined( _MSC_VER )
        return __popcnt( a );
#else
        uint8_t result = 0;
        while ( a )
        {
            result += a & 1;
            a = a >> 1;
        }
        return a;
#endif
    }

    template<>
    inline APO_HOST APO_DEVICE

        uint32_t
        popcount( uint8_t a )
    {
        return static_cast<uint8_t>( apo::popcount( static_cast<uint32_t>( a ) ) );
    }

    template<class Type>
    APO_HOST APO_DEVICE

        uint32_t
        ffs( uint32_t a )
    {
        static_assert( AlwaysFalse<Type>, "ffs is not implemented for this type." );
        return 0;
    }

    template<>
    inline APO_HOST APO_DEVICE

        uint32_t
        ffs( uint32_t a )
    {
#if defined( __GNUC__ ) || defined( __clang__ )
        return __builtin_ffs( a ) - 1;
#elif defined( _MSC_VER )
        unsigned long index;
        _BitScanReverse( &index, a );
        return index;
#else
        uint32_t result = 0;
        while ( ( a & 1 ) == 0 )
        {
            result++;
            a = a >> 1;
        }
        return a;
#endif
    }

    template<>
    inline APO_HOST APO_DEVICE

        uint32_t
        ffs( uint8_t a )
    {
        return static_cast<uint8_t>( apo::ffs( static_cast<uint32_t>( a ) ) );
    }

} // namespace apo
