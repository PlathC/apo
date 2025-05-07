#include "apo/core/type.hpp"

namespace apo
{
    template<class Type>
    Span<Type>::Span( Type * pptr, std::size_t psize ) : ptr( pptr ), size( psize )
    {
    }

    template<class Type>
    Span<Type>::Span( std::vector<Type> & data ) : ptr( data.data() ), size( data.size() )
    {
    }

    template<class Type>
    template<std::size_t Size>
    Span<Type>::Span( std::array<Type, Size> & data ) : ptr( data.data() ), size( Size )
    {
    }

    template<class Type>
    Type & Span<Type>::operator[]( std::size_t i ) const
    {
        return *( ptr + i );
    }

    template<class Type>
    Type * Span<Type>::begin() const
    {
        return ptr;
    }

    template<class Type>
    Type * Span<Type>::end() const
    {
        return ptr + size;
    }

    template<class Type>
    ConstSpan<Type>::ConstSpan( const Type * pptr, std::size_t psize ) : ptr( pptr ), size( psize )
    {
    }

    template<class Type>
    ConstSpan<Type>::ConstSpan( const std::vector<Type> & data ) : ptr( data.data() ), size( data.size() )
    {
    }

    template<class Type>
    template<std::size_t Size>
    ConstSpan<Type>::ConstSpan( const std::array<Type, Size> & data ) : ptr( data.data() ), size( Size )
    {
    }

    template<class Type>
    ConstSpan<Type>::ConstSpan( const Span<Type> data ) : ptr( data.ptr ), size( data.size )
    {
    }

    template<class Type>
    const Type & ConstSpan<Type>::operator[]( std::size_t i ) const
    {
        return *( ptr + i );
    }

    template<class Type>
    const Type * ConstSpan<Type>::begin() const
    {
        return ptr;
    }

    template<class Type>
    const Type * ConstSpan<Type>::end() const
    {
        return ptr + size;
    }
} // namespace apo