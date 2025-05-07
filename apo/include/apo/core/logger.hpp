#ifndef APO_CORE_LOGGER_HPP
#define APO_CORE_LOGGER_HPP

#include <fmt/core.h>

namespace apo::logger
{
    enum class Level
    {
        Info,
        Debug,
        Warning,
        Error
    };

    void log( const Level level, std::string_view str );

    template<typename... T>
    void log( const Level level, std::string_view str, T &&... args );

    template<typename... T>
    void debug( std::string_view str, T &&... args );

    template<typename... T>
    void info( std::string_view str, T &&... args );

    template<typename... T>
    void warning( std::string_view str, T &&... args );

    template<typename... T>
    void error( std::string_view str, T &&... args );
} // namespace apo::logger

#include "apo/core/logger.inl"

#endif // APO_CORE_LOGGER_HPP
