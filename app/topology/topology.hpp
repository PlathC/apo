#ifndef APO_TOPOLOGY_TOPOLOGY_HPP
#define APO_TOPOLOGY_TOPOLOGY_HPP

#include <vector>

#include <apo/gpu/algorithm.hpp>

namespace apo
{
    std::vector<char> toCsv( const apo::gpu::Topology & topology );
} // namespace apo

#endif // APO_TOPOLOGY_TOPOLOGY_HPP