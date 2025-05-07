#ifndef APO_GPU_UTILS_CUH
#define APO_GPU_UTILS_CUH

#include <cstdint>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "apo/gpu/setup.hpp"

namespace apo::gpu
{
    inline APO_DEVICE uint8_t clz( uint32_t v ) { return __clz( v ); }
    inline APO_DEVICE uint8_t clz( uint64_t v ) { return __clzll( v ); }

    template<class Mask>
    APO_DEVICE uint8_t getIdFromMask( Mask mask )
    {
        return ( sizeof( Mask ) * 8 - 1u ) - clz( mask );
    }

    template<class Mask, class Predicate>
    APO_DEVICE void forEach( Mask mask, Predicate predicate )
    {
        while ( mask )
        {
            const uint8_t id = getIdFromMask( mask );
            predicate( id );
            mask &= ~( 1u << id );
        }
    }

    template<class Type, uint32_t WarpSize = 32>
    APO_DEVICE Type warpSum( cg::thread_block_tile<WarpSize> warp, Type value )
    {
        for ( uint32_t i = 1; i < WarpSize; i *= 2 )
            value += warp.shfl_xor( value, i );

        return value;
    }

    template<class Type, uint32_t WarpSize = 32, class Predicate>
    APO_DEVICE Type warpSum( cg::thread_block_tile<WarpSize> warp, Predicate predicate, uint32_t size )
    {
        const uint32_t stepNb = ( size / WarpSize ) + 1;
        Type           sum    = Type( 0 );
        for ( uint32_t s = 0; s < stepNb; s++ )
        {
            const uint32_t id = warp.thread_rank() + warpSize * s;
            sum += warpSum<Type, WarpSize>( warp, predicate( id ) );
        }

        return sum;
    }

    template<class Type, uint32_t WarpSize = 32>
    APO_DEVICE Type warpExclusiveScan( cg::thread_block_tile<WarpSize> warp, Type value )
    {
        Type v = value;
        for ( uint8_t id = 1; id < WarpSize; id *= 2 )
        {
            const Type otherValue = warp.shfl_up( v, id );
            if ( warp.thread_rank() >= id )
                v += otherValue;
        }

        return v - value;
    }

    template<class Type, uint32_t WarpSize = 32>
    APO_DEVICE Type warpInclusiveScan( cg::thread_block_tile<WarpSize> warp, Type value )
    {
        Type v = value;
        for ( uint8_t id = 1; id < WarpSize; id *= 2 )
        {
            const Type otherValue = warp.shfl_up( v, id );
            if ( warp.thread_rank() >= id )
                v += otherValue;
        }

        return v;
    }

    template<class Type, uint32_t WarpSize = 32>
    APO_DEVICE Type warpMax( cg::thread_block_tile<WarpSize> warp, Type value )
    {
        // https://people.maths.ox.ac.uk/~gilesm/cuda/lecs/lec4.pdf
        for ( uint32_t i = 1; i < WarpSize; i *= 2 )
            value = apo::max( value, warp.shfl_xor( value, i ) );

        return value;
    }

    template<class Type, uint32_t WarpSize = 32>
    APO_DEVICE Type warpMin( cg::thread_block_tile<WarpSize> warp, Type value )
    {
        // https://people.maths.ox.ac.uk/~gilesm/cuda/lecs/lec4.pdf
        for ( uint32_t i = 1; i < WarpSize; i *= 2 )
            value = apo::min( value, warp.shfl_xor( value, i ) );

        return value;
    }

    template<uint32_t WarpSize = 32, class Type, class Predicate>
    APO_DEVICE uint32_t
    warpCompact( cg::thread_block_tile<WarpSize> warp, Type * buffer, uint32_t size, Predicate predicate )
    {
        const uint32_t stepNb    = ( size / WarpSize ) + 1;
        uint32_t       removedNb = 0;
        for ( uint32_t s = 0; s < stepNb; s++ )
        {
            const uint32_t id       = warp.thread_rank() + warpSize * s;
            const Type     data     = buffer[ id ];
            const bool     toRemove = predicate( id, data );

            // Count number of removed data before the current one
            const uint32_t toRemoveData = warp.ballot( toRemove );
            const uint8_t  offset       = __popc( toRemoveData << ( 32 - warp.thread_rank() ) );

            // Write vertex to new position
            warp.sync();
            if ( !toRemove )
                buffer[ id - removedNb - offset ] = data;

            removedNb += __popc( toRemoveData );
        }

        return removedNb;
    }

    template<class Type, uint32_t WarpSize = 32>
    __device__ Type warpInclusiveMax( cg::thread_block_tile<WarpSize> warp, Type value )
    {
        for ( uint8_t id = 1; id < WarpSize; id *= 2 )
        {
            const Type otherValue = warp.shfl_up( value, id );
            if ( warp.thread_rank() >= id )
                value = max( otherValue, value );
        }

        return value;
    }

    // Reference: https://www.mgarland.org/files/papers/nvr-2008-003.pdf
    template<class Type, uint32_t WarpSize = 32>
    __device__ Type warpInclusiveSegmentedScan( cg::thread_block_tile<WarpSize> warp, Type value, bool flag )
    {
        const bool initialFlag = flag;
        const int  minIndex    = warpInclusiveMax<int, 32>( warp, flag ? warp.thread_rank() : 0 );
        for ( uint8_t id = 1; id < WarpSize; id *= 2 )
        {
            const Type otherValue = warp.shfl_up( value, id );
            if ( warp.thread_rank() >= minIndex + id )
                value = otherValue + value;
        }

        return value;
    }
} // namespace apo::gpu

#endif // APO_GPU_UTILS_CUH