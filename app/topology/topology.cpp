#include "topology.hpp"

#include <fstream>

#include <apo/core/utils.hpp>
#include <apo/gpu/algorithm.hpp>
#include <common/voronota.hpp>

namespace apo
{
    std::vector<char> toCsv( const apo::gpu::Topology & topology )
    {
        std::vector<char> out {};
        fmt::format_to( std::back_inserter( out ), "{{\n" );
        fmt::format_to( std::back_inserter( out ), "\t\"bisectors\": [\n" );
        const uint32_t bisectorNb = topology.bisectors.size() / 2;
        for ( uint32_t b = 0; b < bisectorNb; b++ )
        {
            const uint32_t i = topology.bisectors[ b * 2 + 0 ];
            const uint32_t j = topology.bisectors[ b * 2 + 1 ];
            fmt::format_to( std::back_inserter( out ), "\t\t[ {}, {} ]{}\n", i, j, b == bisectorNb - 1 ? "" : "," );
        }
        fmt::format_to( std::back_inserter( out ), "\t],\n" );

        fmt::format_to( std::back_inserter( out ), "\t\"trisectors\": [\n" );
        const uint32_t trisectorNb = topology.trisectors.size() / 3;
        for ( uint32_t t = 0; t < trisectorNb; t++ )
        {
            const uint32_t i = topology.trisectors[ t * 3 + 0 ];
            const uint32_t j = topology.trisectors[ t * 3 + 1 ];
            const uint32_t k = topology.trisectors[ t * 3 + 2 ];
            fmt::format_to(
                std::back_inserter( out ), "\t\t[ {}, {}, {} ]{}\n", i, j, k, t == trisectorNb - 1 ? "" : "," );
        }
        fmt::format_to( std::back_inserter( out ), "\t],\n" );

        fmt::format_to( std::back_inserter( out ), "\t\"quadrisectors\": [\n" );
        const uint32_t quadrisectorNb = topology.quadrisectors.size() / 4;
        for ( uint32_t q = 0; q < quadrisectorNb; q++ )
        {
            const uint32_t i = topology.quadrisectors[ q * 4 + 0 ];
            const uint32_t j = topology.quadrisectors[ q * 4 + 1 ];
            const uint32_t k = topology.quadrisectors[ q * 4 + 2 ];
            const uint32_t l = topology.quadrisectors[ q * 4 + 3 ];
            fmt::format_to( std::back_inserter( out ),
                            "\t\t[ {}, {}, {}, {} ]{}\n",
                            i,
                            j,
                            k,
                            l,
                            q == quadrisectorNb - 1 ? "" : "," );
        }
        fmt::format_to( std::back_inserter( out ), "\t]\n" );
        fmt::format_to( std::back_inserter( out ), "}}\n" );

        return out;
    }
} // namespace apo
