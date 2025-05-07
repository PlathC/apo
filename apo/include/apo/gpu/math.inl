#include "apo/gpu/math.cuh"

namespace apo::gpu
{
    inline APO_HOST APO_DEVICE Aabbf::Aabbf( float3 mmin, float3 mmax ) : min( mmin ), max( mmax ) {}
    inline APO_HOST APO_DEVICE Aabbf::Aabbf( float3 point ) : min( point ), max( point ) {}
    inline APO_HOST APO_DEVICE Aabbf::Aabbf( float4 sphere ) :
        min( make_float3( sphere ) - sphere.w ), max( make_float3( sphere ) + sphere.w )
    {
    }

    inline APO_DEVICE APO_HOST Aabbf Aabbf::Degenerated()
    {
        const float maxValue    = MaxFloat;
        const float lowestValue = -MaxFloat;
        return { float3 { maxValue, maxValue, maxValue }, float3 { lowestValue, lowestValue, lowestValue } };
    }

    inline APO_DEVICE APO_HOST float3 Aabbf::getCentroid() const { return ( min + max ) * .5f; }
    inline APO_DEVICE APO_HOST float3 Aabbf::getRadius() const
    {
        return float3 {
            apo::abs( min.x - max.x ),
            apo::abs( min.y - max.y ),
            apo::abs( min.z - max.z ),
        } * .5f;
    }

    inline APO_DEVICE APO_HOST void Aabbf::expand( float3 v )
    {
        min = float3 { apo::min( v.x, min.x ), apo::min( v.y, min.y ), apo::min( v.z, min.z ) };
        max = float3 { apo::max( v.x, max.x ), apo::max( v.y, max.y ), apo::max( v.z, max.z ) };
    }

    inline APO_DEVICE APO_HOST bool Aabbf::isIn( float3 v ) const
    {
        return apo::greaterThan( v.x, min.x ) && apo::greaterThan( v.y, min.y ) && apo::greaterThan( v.z, min.z ) //
               && apo::lessThan( v.x, max.x ) && apo::lessThan( v.y, max.y ) && apo::lessThan( v.z, max.z );
    }

    inline APO_DEVICE APO_HOST bool Aabbf::isIn( float4 sphere ) const
    {
        const float3 center { sphere.x, sphere.y, sphere.z };
        return apo::greaterThan( center.x - sphere.w, min.x )    //
               && apo::greaterThan( center.y - sphere.w, min.y ) //
               && apo::greaterThan( center.z - sphere.w, min.z ) //
               && apo::lessThan( center.x + sphere.w, max.x )    //
               && apo::lessThan( center.y + sphere.w, max.y )    //
               && apo::lessThan( center.z + sphere.w, max.z );   //
    }

    inline APO_DEVICE APO_HOST bool Aabbf::intersect( const float4 sphere ) const
    {
        // Reference: https://github.com/erich666/GraphicsGems/blob/master/gems/BoxSphere.c#L92
        float r2   = sphere.w * sphere.w;
        float dMin = 0.f;

#define intersect_update( v, mi, ma ) \
    if ( v < mi )                     \
        dMin += apo::pow2( v - mi );  \
    else if ( v > ma )                \
        dMin += apo::pow2( v - ma );

        intersect_update( sphere.x, min.x, max.x );
        intersect_update( sphere.y, min.y, max.y );
        intersect_update( sphere.z, min.z, max.z );
#undef intersect_update

        return apo::lessThan( dMin, r2 );
    }

    inline APO_DEVICE APO_HOST bool Aabbf::intersect( const Aabbf other ) const
    {
        return ( apo::lessThan( min.x, other.max.x ) && apo::greaterThan( max.x, other.min.x ) ) && //
               ( apo::lessThan( min.y, other.max.y ) && apo::greaterThan( max.y, other.min.y ) ) && //
               ( apo::lessThan( min.z, other.max.z ) && apo::greaterThan( max.z, other.min.z ) );
    }

    inline APO_HOST APO_DEVICE float3 Aabbf::closestPoint( float4 sphere ) const
    {
        const float3 center = make_float3( sphere );
        return fminf( fmaxf( center, min ), max );
    }

    inline APO_HOST APO_DEVICE float Aabbf::distance( float4 sphere ) const
    {
        return length( make_float3( sphere ) - closestPoint( sphere ) ) - sphere.w;
    }

    inline APO_HOST APO_DEVICE float Aabbf::farthestDistance( float4 sphere ) const
    {
        const float3 center = make_float3( sphere ) - getCentroid();
        const float3 radius = getRadius();

        const float3 farthestPoint = float3 {
            radius.x * apo::sign( -center.x ),
            radius.y * apo::sign( -center.y ),
            radius.z * apo::sign( -center.z ),
        };
        return length( center - farthestPoint ) - sphere.w;
    }

    inline APO_DEVICE APO_HOST uint8_t Aabbf::getLargestAxis() const
    {
        const float x = max.x - min.x;
        const float y = max.y - min.y;
        const float z = max.z - min.z;

        uint8_t axis = 0;
        if ( y > x && y > z )
            axis = 1;
        else if ( z > x && z > y )
            axis = 2;

        return axis;
    }
} // namespace apo::gpu