#include "precision.hpp"

#include <array>

#include <boost/numeric/interval.hpp>

namespace apo::precision
{
    using interval = boost::numeric::interval<Real>;

    using Point = std::array<double, 3>;
    using Site  = std::array<double, 4>;

    interval weightedDistance( const Site si, const Point p )
    {
        interval res = 0;
        for ( uint8_t i = 0; i < 3; ++i )
        {
            interval diff = interval( si[ i ] ) - interval( p[ i ] );
            res += diff * diff;
        }

        return sqrt( res ) - interval( si[ 3 ] );
    }

    std::size_t getBadQuadrisectorNb( const apo::ConstSpan<Real>        sites,
                                      const std::vector<Quadrisector> & quadrisectors )
    {
        const std::size_t siteNb               = sites.size / 4;
        std::size_t       vertexDoesNotExistNb = 0;
        for ( std::size_t q = 0; q < quadrisectors.size(); q++ )
        {
            const Quadrisector & quadrisector = quadrisectors[ q ];
            const Point          point        = { quadrisector.x, quadrisector.y, quadrisector.z };

            const Site si = {
                sites[ quadrisector.i * 4 + 0 ],
                sites[ quadrisector.i * 4 + 1 ],
                sites[ quadrisector.i * 4 + 2 ],
                sites[ quadrisector.i * 4 + 3 ],
            };
            const interval qiRadius = weightedDistance( si, point );

            const Site sj = {
                sites[ quadrisector.j * 4 + 0 ],
                sites[ quadrisector.j * 4 + 1 ],
                sites[ quadrisector.j * 4 + 2 ],
                sites[ quadrisector.j * 4 + 3 ],
            };
            const interval qjRadius = weightedDistance( sj, point );

            const Site sk = {
                sites[ quadrisector.k * 4 + 0 ],
                sites[ quadrisector.k * 4 + 1 ],
                sites[ quadrisector.k * 4 + 2 ],
                sites[ quadrisector.k * 4 + 3 ],
            };
            const interval qkRadius = weightedDistance( sk, point );

            const Site sl = {
                sites[ quadrisector.l * 4 + 0 ],
                sites[ quadrisector.l * 4 + 1 ],
                sites[ quadrisector.l * 4 + 2 ],
                sites[ quadrisector.l * 4 + 3 ],
            };
            const interval qlRadius = weightedDistance( sl, point );

            // Max radius to assert that the point is owned by the four quadrisector sites
            const interval qRadiusMax      = max( qiRadius, max( qjRadius, max( qkRadius, qlRadius ) ) );
            bool           foundCloserSite = false;
            {
                using namespace boost::numeric::interval_lib::compare::certain;
                for ( std::size_t s = 0; s < siteNb; s++ )
                {
                    if ( s == quadrisector.i || s == quadrisector.j || s == quadrisector.k || s == quadrisector.l )
                        continue;

                    const Site site = {
                        sites[ s * 4 + 0 ],
                        sites[ s * 4 + 1 ],
                        sites[ s * 4 + 2 ],
                        sites[ s * 4 + 3 ],
                    };
                    const interval distance = weightedDistance( site, point );
                    if ( distance < qRadiusMax )
                    {
                        foundCloserSite = true;
                        break;
                    }
                }
            }

            if ( foundCloserSite )
                vertexDoesNotExistNb++;
        }

        return vertexDoesNotExistNb;
    }

    std::size_t getBadTrisectorNb( const apo::ConstSpan<Real> sites, const std::vector<Trisector> & closedTrisectors )
    {
        const std::size_t siteNb             = sites.size / 4;
        std::size_t       edgeDoesNotExistNb = 0;
        for ( std::size_t t = 0; t < closedTrisectors.size(); t++ )
        {
            const Trisector & trisector = closedTrisectors[ t ];
            const Point       point     = { trisector.x, trisector.y, trisector.z };

            const Site si = {
                sites[ trisector.i * 4 + 0 ],
                sites[ trisector.i * 4 + 1 ],
                sites[ trisector.i * 4 + 2 ],
                sites[ trisector.i * 4 + 3 ],
            };
            const interval qiRadius = weightedDistance( si, point );

            const Site sj = {
                sites[ trisector.j * 4 + 0 ],
                sites[ trisector.j * 4 + 1 ],
                sites[ trisector.j * 4 + 2 ],
                sites[ trisector.j * 4 + 3 ],
            };
            const interval qjRadius = weightedDistance( sj, point );

            const Site sk = {
                sites[ trisector.k * 4 + 0 ],
                sites[ trisector.k * 4 + 1 ],
                sites[ trisector.k * 4 + 2 ],
                sites[ trisector.k * 4 + 3 ],
            };
            const interval qkRadius = weightedDistance( sk, point );

            // Max radius to assert that the point is owned by the three trisectors sites
            const interval tRadiusMax      = max( qiRadius, max( qjRadius, qkRadius ) );
            bool           foundCloserSite = false;
            {
                using namespace boost::numeric::interval_lib::compare::certain;
                for ( std::size_t s = 0; s < siteNb; s++ )
                {
                    if ( s == trisector.i || s == trisector.j || s == trisector.k )
                        continue;

                    const Site site = {
                        sites[ s * 4 + 0 ],
                        sites[ s * 4 + 1 ],
                        sites[ s * 4 + 2 ],
                        sites[ s * 4 + 3 ],
                    };
                    const interval distance = weightedDistance( site, point );
                    if ( distance < tRadiusMax )
                    {
                        foundCloserSite = true;
                        break;
                    }
                }
            }

            if ( foundCloserSite )
                edgeDoesNotExistNb++;
        }

        return edgeDoesNotExistNb;
    }

} // namespace apo::precision
