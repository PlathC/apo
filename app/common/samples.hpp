#ifndef APO_APP_COMMON_SAMPLES_HPP
#define APO_APP_COMMON_SAMPLES_HPP

#include <filesystem>

#include <apo/core/logger.hpp>
#include <apo/core/math.hpp>
#include <apo/core/utils.hpp>

namespace apo
{
    using Path = std::filesystem::path;

    std::vector<apo::Real> loadProtein( const apo::Path & path );
    std::vector<apo::Real> getUniform( uint32_t  count,
                                       apo::Real spreading,
                                       apo::Real radiiFactor,
                                       apo::Real radiiStart );

    std::vector<apo::Real> parseFromDataset( const apo::Path & path );
} // namespace apo

#endif // APO_APP_COMMON_SAMPLES_HPP