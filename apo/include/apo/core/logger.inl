#include "apo/core/logger.hpp"

namespace apo::logger
{
    template<typename... T>
    void log( const Level level, std::string_view str, T &&... args )
    {
        log( level, fmt::format( str, args... ) );
    }

    template<typename... T>
    void debug( std::string_view str, T &&... args )
    {
#ifndef NDEBUG
        log( Level::Debug, str, args... );
#endif // NDEBUG
    }

    template<typename... T>
    void info( std::string_view str, T &&... args )
    {
        log( Level::Info, str, args... );
    }

    template<typename... T>
    void warning( std::string_view str, T &&... args )
    {
        log( Level::Warning, str, args... );
    }

    template<typename... T>
    void error( std::string_view str, T &&... args )
    {
        log( Level::Error, str, args... );
    }
} // namespace apo::logger