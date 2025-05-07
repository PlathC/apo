#ifndef APO_GPU_HEAP_HPP
#define APO_GPU_HEAP_HPP

#include "apo/gpu/setup.hpp"

namespace apo::gpu
{
    // From Restricted Power Diagrams on the GPU
    // By Justine Basselin, Laurent Alonso, Nicolas Ray, Dmitry Sokolov, Sylvain Lefebvre, Bruno Lévy
    // https://github.com/basselin7u/GPU-Restricted-Power-Diagrams/blob/196e7d79125fb7edc75ed7b8f13990a7877c48f3/knearests.cl#L12
    template<class Float>
    inline APO_HOST APO_DEVICE void heapify( uint32_t * keys, Float * values, uint32_t node, uint32_t size )
    {
        uint32_t j = node;
        while ( true )
        {
            uint32_t left    = 2 * j + 1;
            uint32_t right   = 2 * j + 2;
            uint32_t largest = j;
            if ( left < size && ( values[ left ] > values[ largest ] ) )
                largest = left;
            if ( right < size && ( values[ right ] > values[ largest ] ) )
                largest = right;

            if ( largest == j )
                return;

            apo::gpu::sswap( values[ j ], values[ largest ] );
            apo::gpu::sswap( keys[ j ], keys[ largest ] );
            j = largest;
        }
    }

    template<class Float>
    inline APO_HOST APO_DEVICE void heapsort( uint32_t * keys, Float * values, int size )
    {
        while ( size )
        {
            apo::gpu::sswap( values[ 0 ], values[ size - 1 ] );
            apo::gpu::sswap( keys[ 0 ], keys[ size - 1 ] );
            heapify<Float>( keys, values, 0, --size );
        }
    }
} // namespace apo::gpu

#endif // APO_GPU_HEAP_HPP