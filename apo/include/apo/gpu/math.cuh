#ifndef APO_GPU_MATH_CUH
#define APO_GPU_MATH_CUH

#include <helper_math.h>
#include <vector_functions.h>

#include "apo/core/math.hpp"

namespace apo::gpu
{
#if APO_REAL_SIZE == 8
    using Real2 = double2;
    using Real3 = double3;
    using Real4 = double4;

    inline APO_DEVICE APO_HOST float3 toFloat( Real3 v ) { return make_float3( v.x, v.y, v.z ); }
    inline APO_DEVICE APO_HOST float4 toFloat( Real4 v ) { return make_float4( v.x, v.y, v.z, v.w ); }

    inline APO_DEVICE APO_HOST Real3 toReal( double3 v ) { return v; }
    inline APO_DEVICE APO_HOST Real3 toReal( float3 v ) { return make_double3( v.x, v.y, v.z ); }
    inline APO_DEVICE APO_HOST Real4 toReal( double4 v ) { return v; }
    inline APO_DEVICE APO_HOST Real4 toReal( float4 v ) { return make_double4( v.x, v.y, v.z, v.w ); }
#else
    using Real2 = float2;
    using Real3 = float3;
    using Real4 = float4;

    inline APO_DEVICE APO_HOST float3 toFloat( Real3 v ) { return v; }
    inline APO_DEVICE APO_HOST float4 toFloat( Real4 v ) { return v; }

    inline APO_DEVICE APO_HOST Real3 toReal( float3 v ) { return v; }
    inline APO_DEVICE APO_HOST Real3 toReal( double3 v ) { return make_float3( v.x, v.y, v.z ); }
    inline APO_DEVICE APO_HOST Real4 toReal( float4 v ) { return v; }
    inline APO_DEVICE APO_HOST Real4 toReal( double4 v ) { return make_float4( v.x, v.y, v.z, v.w ); }
#endif

    // double library
    inline __host__ __device__ double2 make_double2( double s ) { return ::make_double2( s, s ); }
    inline __host__ __device__ double2 make_double2( double3 a ) { return ::make_double2( a.x, a.y ); }
    inline __host__ __device__ double2 make_double2( int2 a ) { return ::make_double2( double( a.x ), double( a.y ) ); }
    inline __host__ __device__ double2 make_double2( uint2 a )
    {
        return ::make_double2( double( a.x ), double( a.y ) );
    }

    inline __host__ __device__ double3 make_double3( double s ) { return ::make_double3( s, s, s ); }
    inline __host__ __device__ double3 make_double3( double2 a ) { return ::make_double3( a.x, a.y, 0.0f ); }
    inline __host__ __device__ double3 make_double3( double2 a, double s ) { return ::make_double3( a.x, a.y, s ); }
    inline __host__ __device__ double3 make_double3( double4 a ) { return ::make_double3( a.x, a.y, a.z ); }
    inline __host__ __device__ double3 make_double3( int3 a )
    {
        return ::make_double3( double( a.x ), double( a.y ), double( a.z ) );
    }
    inline __host__ __device__ double3 make_double3( uint3 a )
    {
        return ::make_double3( double( a.x ), double( a.y ), double( a.z ) );
    }

    inline __host__ __device__ double4 make_double4( double s ) { return ::make_double4( s, s, s, s ); }
    inline __host__ __device__ double4 make_double4( double3 a ) { return ::make_double4( a.x, a.y, a.z, 0.0f ); }
    inline __host__ __device__ double4 make_double4( double3 a, double w )
    {
        return ::make_double4( a.x, a.y, a.z, w );
    }
    inline __host__ __device__ double4 make_double4( int4 a )
    {
        return ::make_double4( double( a.x ), double( a.y ), double( a.z ), double( a.w ) );
    }
    inline __host__ __device__ double4 make_double4( uint4 a )
    {
        return ::make_double4( double( a.x ), double( a.y ), double( a.z ), double( a.w ) );
    }

    inline __host__ __device__ double2 operator-( const double2 & a ) { return ::make_double2( -a.x, -a.y ); }
    inline __host__ __device__ double3 operator-( const double3 & a ) { return ::make_double3( -a.x, -a.y, -a.z ); }
    inline __host__ __device__ double4 operator-( const double4 & a )
    {
        return ::make_double4( -a.x, -a.y, -a.z, -a.w );
    }
    inline __host__ __device__ double2 operator-( const double2 & a, const double2 & b )
    {
        return ::make_double2( a.x - b.x, a.y - b.y );
    }
    inline __host__ __device__ double3 operator-( const double3 & a, const double3 & b )
    {
        return ::make_double3( a.x - b.x, a.y - b.y, a.z - b.z );
    }
    inline __host__ __device__ double4 operator-( const double4 & a, const double4 & b )
    {
        return ::make_double4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
    }
    inline __host__ __device__ double2 operator-( const double2 & a, double b )
    {
        return ::make_double2( a.x - b, a.y - b );
    }
    inline __host__ __device__ double3 operator-( const double3 & a, double b )
    {
        return ::make_double3( a.x - b, a.y - b, a.z - b );
    }
    inline __host__ __device__ double4 operator-( const double4 & a, double b )
    {
        return ::make_double4( a.x - b, a.y - b, a.z - b, a.w - b );
    }
    inline __host__ __device__ double2 operator-( double b, const double2 & a )
    {
        return ::make_double2( a.x - b, a.y - b );
    }
    inline __host__ __device__ double3 operator-( double b, const double3 & a )
    {
        return ::make_double3( a.x - b, a.y - b, a.z - b );
    }
    inline __host__ __device__ double4 operator-( double b, const double4 & a )
    {
        return ::make_double4( a.x - b, a.y - b, a.z - b, a.w - b );
    }

    inline __host__ __device__ double2 operator+( const double2 & a, const double2 & b )
    {
        return ::make_double2( a.x + b.x, a.y + b.y );
    }
    inline __host__ __device__ double3 operator+( const double3 & a, const double3 & b )
    {
        return ::make_double3( a.x + b.x, a.y + b.y, a.z + b.z );
    }
    inline __host__ __device__ double4 operator+( const double4 & a, const double4 & b )
    {
        return ::make_double4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
    }
    inline __host__ __device__ double2 operator+( const double2 & a, double b )
    {
        return ::make_double2( a.x + b, a.y + b );
    }
    inline __host__ __device__ double3 operator+( const double3 & a, double b )
    {
        return ::make_double3( a.x + b, a.y + b, a.z + b );
    }
    inline __host__ __device__ double4 operator+( const double4 & a, double b )
    {
        return ::make_double4( a.x + b, a.y + b, a.z + b, a.w + b );
    }
    inline __host__ __device__ double2 operator+( double b, const double2 & a )
    {
        return ::make_double2( a.x + b, a.y + b );
    }
    inline __host__ __device__ double3 operator+( double b, const double3 & a )
    {
        return ::make_double3( a.x + b, a.y + b, a.z + b );
    }
    inline __host__ __device__ double4 operator+( double b, const double4 & a )
    {
        return ::make_double4( a.x + b, a.y + b, a.z + b, a.w + b );
    }

    inline __host__ __device__ void operator+=( double2 & a, double2 b )
    {
        a.x += b.x;
        a.y += b.y;
    }

    inline __host__ __device__ double3 operator*( double b, double3 a )
    {
        return ::make_double3( b * a.x, b * a.y, b * a.z );
    }
    inline __host__ __device__ double4 operator*( double4 a, double4 b )
    {
        return ::make_double4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
    }

    inline __host__ __device__ double3 operator*( double3 a, double b )
    {
        return ::make_double3( b * a.x, b * a.y, b * a.z );
    }

    inline __host__ __device__ void operator*=( double3 & a, double b )
    {
        a.x *= b;
        a.y *= b;
        a.z *= b;
    }
    inline __host__ __device__ void operator*=( double4 & a, double4 b )
    {
        a.x *= b.x;
        a.y *= b.y;
        a.z *= b.z;
        a.w *= b.w;
    }
    inline __host__ __device__ double4 operator*( double4 a, double b )
    {
        return ::make_double4( a.x * b, a.y * b, a.z * b, a.w * b );
    }
    inline __host__ __device__ double4 operator*( double b, double4 a )
    {
        return ::make_double4( b * a.x, b * a.y, b * a.z, b * a.w );
    }
    inline __host__ __device__ void operator*=( double4 & a, double b )
    {
        a.x *= b;
        a.y *= b;
        a.z *= b;
        a.w *= b;
    }

    inline __host__ __device__ double2 operator/( double2 a, double2 b )
    {
        return ::make_double2( a.x / b.x, a.y / b.y );
    }
    inline __host__ __device__ void operator/=( double2 & a, double2 b )
    {
        a.x /= b.x;
        a.y /= b.y;
    }
    inline __host__ __device__ double2 operator/( double2 a, double b ) { return ::make_double2( a.x / b, a.y / b ); }
    inline __host__ __device__ void    operator/=( double2 & a, double b )
    {
        a.x /= b;
        a.y /= b;
    }
    inline __host__ __device__ double2 operator/( double b, double2 a ) { return ::make_double2( b / a.x, b / a.y ); }

    inline __host__ __device__ double3 operator/( double3 a, double3 b )
    {
        return ::make_double3( a.x / b.x, a.y / b.y, a.z / b.z );
    }
    inline __host__ __device__ void operator/=( double3 & a, double3 b )
    {
        a.x /= b.x;
        a.y /= b.y;
        a.z /= b.z;
    }
    inline __host__ __device__ double3 operator/( double3 a, double b )
    {
        return ::make_double3( a.x / b, a.y / b, a.z / b );
    }
    inline __host__ __device__ void operator/=( double3 & a, double b )
    {
        a.x /= b;
        a.y /= b;
        a.z /= b;
    }
    inline __host__ __device__ double3 operator/( double b, double3 a )
    {
        return ::make_double3( b / a.x, b / a.y, b / a.z );
    }

    inline __host__ __device__ double4 operator/( double4 a, double4 b )
    {
        return ::make_double4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
    }
    inline __host__ __device__ void operator/=( double4 & a, double4 b )
    {
        a.x /= b.x;
        a.y /= b.y;
        a.z /= b.z;
        a.w /= b.w;
    }
    inline __host__ __device__ double4 operator/( double4 a, double b )
    {
        return ::make_double4( a.x / b, a.y / b, a.z / b, a.w / b );
    }
    inline __host__ __device__ void operator/=( double4 & a, double b )
    {
        a.x /= b;
        a.y /= b;
        a.z /= b;
        a.w /= b;
    }
    inline __host__ __device__ double4 operator/( double b, double4 a )
    {
        return ::make_double4( b / a.x, b / a.y, b / a.z, b / a.w );
    }

    inline __device__ __host__ double clamp( double f, double a, double b ) { return max( a, min( f, b ) ); }

    inline __device__ __host__ double2 clamp( double2 v, double a, double b )
    {
        return ::make_double2( clamp( v.x, a, b ), clamp( v.y, a, b ) );
    }
    inline __device__ __host__ double2 clamp( double2 v, double2 a, double2 b )
    {
        return ::make_double2( clamp( v.x, a.x, b.x ), clamp( v.y, a.y, b.y ) );
    }
    inline __device__ __host__ double3 clamp( double3 v, double a, double b )
    {
        return ::make_double3( clamp( v.x, a, b ), clamp( v.y, a, b ), clamp( v.z, a, b ) );
    }
    inline __device__ __host__ double3 clamp( double3 v, double3 a, double3 b )
    {
        return ::make_double3( clamp( v.x, a.x, b.x ), clamp( v.y, a.y, b.y ), clamp( v.z, a.z, b.z ) );
    }
    inline __device__ __host__ double4 clamp( double4 v, double a, double b )
    {
        return ::make_double4( clamp( v.x, a, b ), clamp( v.y, a, b ), clamp( v.z, a, b ), clamp( v.w, a, b ) );
    }
    inline __device__ __host__ double4 clamp( double4 v, double4 a, double4 b )
    {
        return ::make_double4(
            clamp( v.x, a.x, b.x ), clamp( v.y, a.y, b.y ), clamp( v.z, a.z, b.z ), clamp( v.w, a.w, b.w ) );
    }

    inline __host__ __device__ double dot( double2 a, double2 b ) { return a.x * b.x + a.y * b.y; }
    inline __host__ __device__ double dot( double3 a, double3 b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    inline __host__ __device__ double dot( double4 a, double4 b )
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    inline __host__ __device__ double length( double2 v ) { return sqrt( dot( v, v ) ); }
    inline __host__ __device__ double length( double3 v ) { return sqrt( dot( v, v ) ); }
    inline __host__ __device__ double length( double4 v ) { return sqrt( dot( v, v ) ); }

    inline __host__ __device__ double3 normalize( double3 v )
    {
        double invLen = rsqrt( dot( v, v ) );
        return v * invLen;
    }
    inline __host__ __device__ double2 abs( double2 v ) { return ::make_double2( ::abs( v.x ), ::abs( v.y ) ); }
    inline __host__ __device__ double3 abs( double3 v )
    {
        return ::make_double3( ::abs( v.x ), ::abs( v.y ), ::abs( v.z ) );
    }
    inline __host__ __device__ double4 abs( double4 v )
    {
        return ::make_double4( ::abs( v.x ), ::abs( v.y ), ::abs( v.z ), ::abs( v.w ) );
    }

    inline __host__ __device__ double3 cross( double3 a, double3 b )
    {
        return ::make_double3( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x );
    }

    // Float library
    inline __host__ __device__ float2 operator-( const float2 & a ) { return ::make_float2( -a.x, -a.y ); }
    inline __host__ __device__ float3 operator-( const float3 & a ) { return ::make_float3( -a.x, -a.y, -a.z ); }
    inline __host__ __device__ float4 operator-( const float4 & a )
    {
        return ::make_float4( -a.x, -a.y, -a.z, -a.w );
    }

    struct Aabbf
    {
        float3 min { 0, 0, 0 };
        float3 max { 0, 0, 0 };

        Aabbf() = default;
        inline APO_HOST APO_DEVICE Aabbf( float3 mmin, float3 mmax );
        inline APO_HOST APO_DEVICE Aabbf( float3 point );
        inline APO_HOST APO_DEVICE Aabbf( float4 sphere );

        APO_HOST APO_DEVICE static Aabbf Degenerated();

        inline APO_HOST APO_DEVICE float3 getCentroid() const;
        inline APO_HOST APO_DEVICE float3 getRadius() const;

        inline APO_HOST APO_DEVICE void   expand( float3 v );
        inline APO_HOST APO_DEVICE bool   isIn( float3 v ) const;
        inline APO_HOST APO_DEVICE bool   isIn( float4 sphere ) const;
        inline APO_HOST APO_DEVICE bool   intersect( float4 sphere ) const;
        inline APO_HOST APO_DEVICE bool   intersect( Aabbf other ) const;
        inline APO_HOST APO_DEVICE float3 closestPoint( float4 sphere ) const;
        inline APO_HOST APO_DEVICE float  distance( float4 sphere ) const;
        inline APO_HOST APO_DEVICE float  farthestDistance( float4 sphere ) const;
        inline APO_HOST APO_DEVICE float  squaredDistance( float4 sphere ) const;
        inline APO_HOST APO_DEVICE float  squaredFarthestDistance( float4 sphere ) const;

        inline APO_HOST APO_DEVICE uint8_t getLargestAxis() const;
    };

    APO_DEVICE APO_HOST inline float sphereDistance( float4 s1, float4 s2 )
    {
        const float distance = length( float3 { s1.x, s1.y, s1.z } - float3 { s2.x, s2.y, s2.z } );
        return distance - ( s1.w + s2.w );
    }

    APO_DEVICE APO_HOST inline double sphereDistance( double4 s1, double4 s2 )
    {
        const double distance = length( double3 { s1.x - s2.x, s1.y - s2.y, s1.z - s2.z } );
        return distance - ( s1.w + s2.w );
    }

    APO_DEVICE APO_HOST inline bool intersect( float4 a, float4 b )
    {
        return apo::lessThan( sphereDistance( a, b ), 0.f );
    }

    APO_DEVICE APO_HOST inline bool intersect( double4 a, double4 b )
    {
        return apo::lessThan( sphereDistance( a, b ), 0. );
    }

    APO_DEVICE inline void inverse( Real2 aaCol1, Real2 aaCol2, Real2 & cc1, Real2 & cc2 )
    {
        const Real invDeterminant = Real( 1 ) / fma( aaCol1.x, aaCol2.y, -aaCol2.x * aaCol1.y );

        cc1 = { invDeterminant * aaCol2.y, -invDeterminant * aaCol1.y };
        cc2 = { -invDeterminant * aaCol2.x, invDeterminant * aaCol1.x };
    }

    APO_DEVICE inline void inverse( Real3 aaCol1, Real3 aaCol2, Real3 aaCol3, Real3 & cc1, Real3 & cc2, Real3 & cc3 )
    {
        // Sarrus's rule
        const Real invD = Real( 1 ) / dot( aaCol1, cross( aaCol2, aaCol3 ) );

        // Transpose of Cofactor matrix
        const Real3 row1 = cross( aaCol2, aaCol3 );
        const Real3 row2 = cross( aaCol3, aaCol1 );
        const Real3 row3 = cross( aaCol1, aaCol2 );

        cc1 = { invD * row1.x, invD * row2.x, invD * row3.x };
        cc2 = { invD * row1.y, invD * row2.y, invD * row3.y };
        cc3 = { invD * row1.z, invD * row2.z, invD * row3.z };
    }

    APO_DEVICE inline void dot( Real2 aaCol1, Real2 aaCol2, Real2 bbCol1, Real2 bbCol2, Real2 & c1, Real2 & c2 )
    {
        c1 = { aaCol1.x * bbCol1.x + aaCol2.x * bbCol1.y, aaCol1.y * bbCol1.x + aaCol2.y * bbCol1.y };
        c2 = { aaCol1.x * bbCol2.x + aaCol2.x * bbCol2.y, aaCol1.y * bbCol2.x + aaCol2.y * bbCol2.y };
    }

    APO_DEVICE inline Real3 dot( Real3 aaCol1, Real3 aaCol2, Real3 aaCol3, Real3 b )
    {
        return Real3 {
            dot( Real3 { aaCol1.x, aaCol2.x, aaCol3.x }, b ),
            dot( Real3 { aaCol1.y, aaCol2.y, aaCol3.y }, b ),
            dot( Real3 { aaCol1.z, aaCol2.z, aaCol3.z }, b ),
        };
    }

    APO_DEVICE inline void eigendecomposition( Real2 mCol1, Real2 mCol2, Real2 & e, Real2 & v )
    {
        // See:
        // https://gensoft.pasteur.fr/docs/lapack/3.9.0/db/db1/dlae2_8f_source.html
        // and
        // https://netlib.org/lapack/explore-html//db/d54/group__laev2_gae1fece521602520c28b76c206104a8d0.html#gae1fece521602520c28b76c206104a8d0
        const Real a = mCol1.x;
        const Real b = mCol2.x;
        const Real c = mCol2.y;

        const Real sm  = a + c;
        const Real df  = a - c;
        const Real adf = ::abs( df );
        const Real tb  = b + b;
        const Real ab  = ::abs( tb );

        Real acmx = a, acmn = c;
        if ( ::abs( a ) <= ::abs( c ) )
            sswap( acmx, acmn );

        Real aa = ab;
        Real bb = adf;
        if ( adf < ab )
            sswap( aa, bb );

        const Real temp = aa / bb;
        Real       rt   = bb * sqrt( Real( 1 ) + temp * temp );
        // This can be safely removed since the above formula results in same
        // computation Real rt = mcb::isClose( aa, bb ) ? ab * SqrtTwo : bb *
        // sqrt( Real( 1 ) + temp * temp );

        Real sgn1 = copysign( Real( 1 ), sm );
        Real rt1  = Real( .5 ) * ( sm + sgn1 * rt );
        Real rt2  = ( acmx / rt1 ) * acmn - ( b / rt1 ) * b;
        /*
            This can be safely removed since the above formula results in same
           computation if ( mcb::isClose( sm, Real( 0 ) ) )
            {
                sgn1 = Real( 1 );
                rt1  = ( .5 ) * rt;
                rt2  = -( .5 ) * rt;
            }
        */

        const Real sgn2 = copysign( Real( 1 ), df );
        const Real cs   = df + sgn2 * rt;

        Real acs = ::abs( cs );
        aa       = tb;
        bb       = cs;
        if ( lessThan( acs, ab ) )
            sswap( aa, bb );

        Real t   = -aa / bb;
        Real sn1 = Real( 1 ) / sqrt( Real( 1 ) + t * t );
        Real cs1 = t * sn1;
        if ( lessThan( acs, ab ) )
            sswap( sn1, cs1 );

        if ( lessThan( acs, ab ) && isCloseToZero( ab ) )
        {
            cs1 = Real( 1 );
            sn1 = Real( 0 );
        }

        if ( isClose( sgn1, sgn2 ) )
        {
            Real tn = cs1;
            cs1     = -sn1;
            sn1     = tn;
        }

        // Write smallest eigenvalue first
        e = { rt2, rt1 };
        v = { -sn1, cs1 };
    }

    APO_DEVICE inline float determinantf( const float4 c1, const float4 c2, const float4 c3, const float4 c4 )
    {
        const float d1
            = dot( float3 { c2.y, c2.z, c2.w }, cross( float3 { c3.y, c3.z, c3.w }, float3 { c4.y, c4.z, c4.w } ) );
        const float d2
            = dot( float3 { c2.x, c2.z, c2.w }, cross( float3 { c3.x, c3.z, c3.w }, float3 { c4.x, c4.z, c4.w } ) );
        const float d3
            = dot( float3 { c2.x, c2.y, c2.w }, cross( float3 { c3.x, c3.y, c3.w }, float3 { c4.x, c4.y, c4.w } ) );
        const float d4
            = dot( float3 { c2.x, c2.y, c2.z }, cross( float3 { c3.x, c3.y, c3.z }, float3 { c4.x, c4.y, c4.z } ) );

        return c1.x * d1 - c1.y * d2 + c1.z * d3 - c1.w * d4;
    }

    APO_DEVICE inline Real determinant( const Real4 c1, const Real4 c2, const Real4 c3, const Real4 c4 )
    {
        const Real d1
            = dot( Real3 { c2.y, c2.z, c2.w }, cross( Real3 { c3.y, c3.z, c3.w }, Real3 { c4.y, c4.z, c4.w } ) );
        const Real d2
            = dot( Real3 { c2.x, c2.z, c2.w }, cross( Real3 { c3.x, c3.z, c3.w }, Real3 { c4.x, c4.z, c4.w } ) );
        const Real d3
            = dot( Real3 { c2.x, c2.y, c2.w }, cross( Real3 { c3.x, c3.y, c3.w }, Real3 { c4.x, c4.y, c4.w } ) );
        const Real d4
            = dot( Real3 { c2.x, c2.y, c2.z }, cross( Real3 { c3.x, c3.y, c3.z }, Real3 { c4.x, c4.y, c4.z } ) );

        return c1.x * d1 - c1.y * d2 + c1.z * d3 - c1.w * d4;
    }

    APO_DEVICE inline Real4 sphereFromPoints( const Real3 p1, const Real3 p2, const Real3 p3, const Real3 p4 )
    {
        const Real t1 = -dot( p1, p1 );
        const Real t2 = -dot( p2, p2 );
        const Real t3 = -dot( p3, p3 );
        const Real t4 = -dot( p4, p4 );

        const Real t = determinant( //
            Real4 { p1.x, p2.x, p3.x, p4.x },
            Real4 { p1.y, p2.y, p3.y, p4.y },
            Real4 { p1.z, p2.z, p3.z, p4.z },
            Real4 { 1, 1, 1, 1 } );

        const Real d = determinant( //
                           Real4 { t1, t2, t3, t4 },
                           Real4 { p1.y, p2.y, p3.y, p4.y },
                           Real4 { p1.z, p2.z, p3.z, p4.z },
                           Real4 { 1, 1, 1, 1 } )
                       / t;
        const Real e = determinant( //
                           Real4 { p1.x, p2.x, p3.x, p4.x },
                           Real4 { t1, t2, t3, t4 },
                           Real4 { p1.z, p2.z, p3.z, p4.z },
                           Real4 { 1, 1, 1, 1 } )
                       / t;
        const Real f = determinant( //
                           Real4 { p1.x, p2.x, p3.x, p4.x },
                           Real4 { p1.y, p2.y, p3.y, p4.y },
                           Real4 { t1, t2, t3, t4 },
                           Real4 { 1, 1, 1, 1 } )
                       / t;
        const Real g = determinant( //
                           Real4 { p1.x, p2.x, p3.x, p4.x },
                           Real4 { p1.y, p2.y, p3.y, p4.y },
                           Real4 { p1.z, p2.z, p3.z, p4.z },
                           Real4 { t1, t2, t3, t4 } )
                       / t;

        return Real4 { -d * Real( .5 ),
                       -e * Real( .5 ),
                       -f * Real( .5 ),
                       Real( .5 ) * ::sqrt( d * d + e * e + f * f - Real( 4. ) * g ) };
    }

    APO_DEVICE inline void getTangentPlanes( const Real4 s1, const Real4 s2, const Real4 s3, Real3 & n1, Real3 & n2 )
    {
        const Real3 p1 = Real3 { s1.x, s1.y, s1.z };
        const Real3 p2 = Real3 { s2.x, s2.y, s2.z };
        const Real3 p3 = Real3 { s3.x, s3.y, s3.z };

        const Real4 nh1 = s1 - s2;
        const Real4 nh2 = s1 - s3;

        const Real2 aC1 = { nh1.x, nh2.x };
        const Real2 aC2 = { nh1.y, nh2.y };
        const Real2 bC1 = { nh1.z, nh2.z };
        const Real2 bC2 = { nh1.w, nh2.w };

        Real2 iAC1, iAC2;
        inverse( aC1, aC2, iAC1, iAC2 );

        Real2 cC1, cC2;
        dot( iAC1, iAC2, bC1, bC2, cC1, cC2 );

        const Real2 eC1 = {
            Real( 1 ) + cC1.x * cC1.x + cC1.y * cC1.y,
            Real( 0 ) + cC1.x * cC2.x + cC1.y * cC2.y,
        };
        const Real2 eC2 = {
            Real( 0 ) + cC1.x * cC2.x + cC1.y * cC2.y,
            Real( -1 ) + cC2.x * cC2.x + cC2.y * cC2.y,
        };

        Real2 eigenvalues, eigenvectors;
        eigendecomposition( eC1, eC2, eigenvalues, eigenvectors );

        const Real alpha0 = ::sqrt( ::abs( eigenvalues.x ) );
        const Real alpha1 = ::sqrt( ::abs( eigenvalues.y ) );

        Real2 z   = { 1, alpha0 / alpha1 };
        Real2 y34 = { dot( Real2 { eigenvectors.x, -eigenvectors.y }, z ),
                      dot( Real2 { eigenvectors.y, eigenvectors.x }, z ) };
        Real  x1  = -dot( Real2 { cC1.x, cC2.x }, y34 );
        Real  x2  = -dot( Real2 { cC1.y, cC2.y }, y34 );
        n1        = normalize( Real3 { x1, x2, y34.x } );

        z   = { 1, -alpha0 / alpha1 };
        y34 = { dot( Real2 { eigenvectors.x, -eigenvectors.y }, z ),
                dot( Real2 { eigenvectors.y, eigenvectors.x }, z ) };
        x1  = -dot( Real2 { cC1.x, cC2.x }, y34 );
        x2  = -dot( Real2 { cC1.y, cC2.y }, y34 );
        n2  = normalize( Real3 { x1, x2, y34.x } );

        const Real3 p = p1 + n1 * s1.w;
        if ( apo::greaterThan( ::abs( dot( n1, p - p1 ) - s1.w ), Real( 1e-4 ) )
             || apo::greaterThan( ::abs( dot( n1, p - p2 ) - s2.w ), Real( 1e-4 ) )
             || apo::greaterThan( ::abs( dot( n1, p - p3 ) - s3.w ), Real( 1e-4 ) ) )
        {
            n1 *= apo::Real( -1 );
            n2 *= apo::Real( -1 );
        }
    }

    APO_DEVICE inline uint8_t trisectorVertex( Real4 s1, Real4 s2, Real4 s3, Real4 & x1, Real4 & x2 )
    {
        const Real3 p1 = { s1.x, s1.y, s1.z };
        const Real3 p2 = { s2.x, s2.y, s2.z };
        const Real3 p3 = { s3.x, s3.y, s3.z };
        const Real  r1 = s1.w;
        const Real  r2 = s2.w;
        const Real  r3 = s3.w;

        const Real4 nh1 = s1 - s2;
        const Real4 nh2 = s1 - s3;

        // Compute P_ijk
        const Real3 nIjk = normalize( cross( p2 - p1, p3 - p1 ) );
        const Real  w    = -dot( nIjk, p1 );

        const Real3 mC {
            apo::Real( .5 ) * ( -dot( p1, p1 ) + dot( p2, p2 ) + r1 * r1 - r2 * r2 ),
            apo::Real( .5 ) * ( -dot( p1, p1 ) + dot( p3, p3 ) + r1 * r1 - r3 * r3 ),
            w,
        };

        const Real3 c1A = { nh1.x, nh2.x, nIjk.x };
        const Real3 c2A = { nh1.y, nh2.y, nIjk.y };
        const Real3 c3A = { nh1.z, nh2.z, nIjk.z };

        Real3 c1iA, c2iA, c3iA;
        inverse( c1A, c2A, c3A, c1iA, c2iA, c3iA );

        const Real3 mN = -dot( c1iA, c2iA, c3iA, Real3 { nh1.w, nh2.w, Real( 0 ) } );
        const Real3 mB = dot( c1iA, c2iA, c3iA, mC );

        const Real e1 = -mN.x * r1 - mB.x - p1.x;
        const Real e2 = -mN.y * r1 - mB.y - p1.y;
        const Real e3 = -mN.z * r1 - mB.z - p1.z;

        const Real a = mN.x * mN.x + mN.y * mN.y + mN.z * mN.z - Real( 1 );
        const Real b = Real( 2 ) * ( mN.x * e1 + mN.y * e2 + mN.z * e3 );
        const Real c = e1 * e1 + e2 * e2 + e3 * e3;

        const Real delta       = b * b - Real( 4 ) * a * c;
        uint8_t    vertexCount = 0;

        const Real q = -Real( .5 ) * ( b - apo::sign( b ) * ::sqrt( delta ) );

        const Real  xd1 = ( c / q ) - r1;
        const Real3 x1n = mN * xd1 - mB;

        if ( apo::greaterThan( xd1 + r1, Real( 0 ) ) )
        {
            x1 = { x1n.x, x1n.y, x1n.z, xd1 };
            vertexCount++;
        }

        const Real  xd12 = ( q / a ) - r1;
        const Real3 x1n2 = mN * xd12 - mB;
        if ( apo::greaterThan( xd12 + r1, apo::Real( 0 ) ) )
        {
            x2 = { x1n2.x, x1n2.y, x1n2.z, xd12 };
            if ( vertexCount == 0 )
                x1 = x2;
            vertexCount++;
        }

        return vertexCount;
    }

    APO_DEVICE inline uint8_t quadrisector( const Real4 & s1,
                                            const Real4 & s2,
                                            const Real4 & s3,
                                            const Real4 & s4,
                                            Real4 &       x1,
                                            Real4 &       x2 )
    {
        const Real3 p1 = { s1.x, s1.y, s1.z };
        const Real3 p2 = { s2.x, s2.y, s2.z };
        const Real3 p3 = { s3.x, s3.y, s3.z };
        const Real3 p4 = { s4.x, s4.y, s4.z };
        const Real  r1 = s1.w;
        const Real  r2 = s2.w;
        const Real  r3 = s3.w;
        const Real  r4 = s4.w;

        const Real4 nh1 = s1 - s2;
        const Real4 nh2 = s1 - s3;
        const Real4 nh3 = s1 - s4;

        const Real3 mC {
            Real( .5 ) * ( -dot( p1, p1 ) + dot( p2, p2 ) + r1 * r1 - r2 * r2 ),
            Real( .5 ) * ( -dot( p1, p1 ) + dot( p3, p3 ) + r1 * r1 - r3 * r3 ),
            Real( .5 ) * ( -dot( p1, p1 ) + dot( p4, p4 ) + r1 * r1 - r4 * r4 ),
        };

        const Real3 c1A = { nh1.x, nh2.x, nh3.x };
        const Real3 c2A = { nh1.y, nh2.y, nh3.y };
        const Real3 c3A = { nh1.z, nh2.z, nh3.z };

        // assert( !apo::isCloseToZero( A.determinant(), apo::Real( 1e-8 ) ) && "Singular matrix" );
        // if ( apo::isCloseToZero( A.determinant(), apo::Real( 1e-8 ) ) )
        //     throw std::runtime_error( "Quadrisector: Singular matrix !" );

        Real3 c1iA, c2iA, c3iA;
        apo::gpu::inverse( c1A, c2A, c3A, c1iA, c2iA, c3iA );

        const Real3 mN = Real3 { 0, 0, 0 } - dot( c1iA, c2iA, c3iA, Real3 { nh1.w, nh2.w, nh3.w } );
        const Real3 mB = dot( c1iA, c2iA, c3iA, mC );

        const Real e1 = -mN.x * r1 - mB.x - p1.x;
        const Real e2 = -mN.y * r1 - mB.y - p1.y;
        const Real e3 = -mN.z * r1 - mB.z - p1.z;

        const Real a = mN.x * mN.x + mN.y * mN.y + mN.z * mN.z - Real( 1 );
        const Real b = Real( 2 ) * ( mN.x * e1 + mN.y * e2 + mN.z * e3 );
        const Real c = e1 * e1 + e2 * e2 + e3 * e3;

        const Real delta = b * b - Real( 4 ) * a * c;
        if ( apo::lessThan( delta, Real( 0 ) ) )
            return 0;

        const Real q = -Real( .5 ) * ( b - apo::sign( b ) * ::sqrt( delta ) );

        const Real  xd1 = ( c / q ) - r1;
        const Real3 x1n = mN * xd1 - mB;
        x1              = { x1n.x, x1n.y, x1n.z, xd1 };
        uint8_t mask    = uint8_t( apo::greaterThan( xd1 + r1, apo::Real( 0 ) ) );

        // It seems that decreasing precision can cause errors on the value of the radius
        // Since it cannot be recovered during validation, recomputation is required
        if ( apo::greaterThan( xd1 + r1, apo::Real( 0 ) ) )
            x1.w = length( p1 - x1n ) - s1.w;

        if ( !apo::isCloseToZero( delta ) )
        {
            const Real  xd12 = ( q / a ) - r1; // ( ( -b + apo::sqrt( delta ) ) / ( Real( 2 ) * a ) ) - r1;
            const Real3 x1n2 = mN * xd12 - mB;
            x2               = { x1n2.x, x1n2.y, x1n2.z, xd12 };

            if ( apo::greaterThan( xd12 + r1, apo::Real( 0 ) ) )
                x2.w = length( p1 - x1n2 ) - s1.w;

            mask |= uint8_t( apo::greaterThan( xd12 + r1, apo::Real( 0 ) ) ) << 1;
        }

        return mask;
    }
} // namespace apo::gpu

#include "apo/gpu/math.inl"

#endif // APO_GPU_MATH_CUH