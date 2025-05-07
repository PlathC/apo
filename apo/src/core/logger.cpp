#include "apo/core/logger.hpp"

#include <fmt/chrono.h>

namespace apo::logger
{
    static std::string_view toString( Level level )
    {
        if ( level == Level::Info )
            return "Info";
        else if ( level == Level::Debug )
            return "Debug";
        else if ( level == Level::Warning )
            return "Warning";
        else if ( level == Level::Error )
            return "Error";

        return "";
    }

    static std::tm now()
    {
        const std::time_t currentTime = std::chrono::system_clock::to_time_t( std::chrono::system_clock::now() );

        // From https://github.com/gabime/spdlog/blob/master/include/spdlog/details/os.h#L73
        std::tm tm;
#ifdef _WIN32
        // https://en.cppreference.com/w/c/chrono/localtime
        // "The implementation of localtime_s in Microsoft CRT is incompatible with the C standard since it has reversed
        // parameter order and returns errno_t."
        ::localtime_s( &tm, &currentTime );
#else
        ::localtime_r( &currentTime, &tm );
#endif

        return tm;
    }

    void log( const Level level, std::string_view str )
    {
        fmt::print( "[{:%Y-%m-%d %H:%M:%S}] [{}] {}\n", now(), toString( level ), str );
    }
} // namespace apo::logger
