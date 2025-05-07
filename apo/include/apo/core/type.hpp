#ifndef APO_CORE_TYPE_HPP
#define APO_CORE_TYPE_HPP

#include <vector>

namespace apo
{
    template<class Type>
    struct Span
    {
        Type *      ptr  = nullptr;
        std::size_t size = 0;

        Span() = default;
        Span( Type * pptr, std::size_t psize );
        Span( std::vector<Type> & data );
        template<std::size_t size>
        Span( std::array<Type, size> & data );

        Type & operator[]( std::size_t i ) const;
        Type * begin() const;
        Type * end() const;
    };

    template<class Type>
    struct ConstSpan
    {
        const Type * ptr  = nullptr;
        std::size_t  size = 0;

        ConstSpan() = default;
        ConstSpan( const Type * pptr, std::size_t psize );
        ConstSpan( const std::vector<Type> & data );
        template<std::size_t size>
        ConstSpan( const std::array<Type, size> & data );
        ConstSpan( const Span<Type> data );

        const Type & operator[]( std::size_t i ) const;
        const Type * begin() const;
        const Type * end() const;
    };
} // namespace apo

#include "apo/core/type.inl"

#endif // APO_CORE_TYPE_HPP
