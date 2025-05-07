#ifndef APO_APP_COMMON_VORONOTA_HPP
#define APO_APP_COMMON_VORONOTA_HPP

#include "../extern/voronota/src/apollota/triangulation.h"
#include "../extern/voronota/src/auxiliaries/io_utilities.h"

// Parallel reference implementation:
// https://github.com/kliment-olechnovic/voronota/blob/c06d48d26479f290c9f62760613cc5a8a1a0158e/src/modes/mode_calculate_vertices_in_parallel.cpp#L15
// Default settings from
// https://github.com/kliment-olechnovic/voronota/blob/c06d48d26479f290c9f62760613cc5a8a1a0158e/src/modes/mode_calculate_vertices_in_parallel.cpp#L379
namespace voronota
{
    struct ParallelComputationResult
    {
        ParallelComputationResult() : number_of_initialized_parts( 0 ), number_of_produced_quadruples( 0 ) {}

        std::vector<voronota::apollota::SimpleSphere>                 input_spheres;
        std::size_t                                                   number_of_initialized_parts;
        std::size_t                                                   number_of_produced_quadruples;
        std::vector<voronota::apollota::Triangulation::QuadruplesMap> distributed_quadruples_maps;
        voronota::apollota::Triangulation::QuadruplesMap              merged_quadruples_map;
    };

    class ParallelComputationProcessingWithOpenMP
    {
      public:
        static void process( ParallelComputationResult & result,
                             const std::size_t           parts                      = 32,
                             const double                init_radius_for_BSH        = 3.5,
                             const bool                  include_surplus_quadruples = false )
        {
            // result.input_spheres.clear();
            std::vector<voronota::apollota::SimpleSphere> & spheres = result.input_spheres;
            // voronota::auxiliaries::IOUtilities().read_lines_to_set(std::cin, spheres);

            const std::vector<std::vector<std::size_t>> distributed_ids
                = voronota::apollota::SplittingOfSpheres::split_for_number_of_parts( spheres, parts );
            result.number_of_initialized_parts = distributed_ids.size();

            const voronota::apollota::BoundingSpheresHierarchy bsh( spheres, init_radius_for_BSH, 1 );

            result.distributed_quadruples_maps
                = std::vector<voronota::apollota::Triangulation::QuadruplesMap>( distributed_ids.size() );
            std::vector<int> distributed_errors( distributed_ids.size(), 0 );

            const int distributed_ids_size = static_cast<int>( distributed_ids.size() );
            {
#pragma omp parallel for
                for ( int i = 0; i < distributed_ids_size; i++ )
                {
                    try
                    {
                        result.distributed_quadruples_maps[ i ]
                            = voronota::apollota::Triangulation::construct_result_for_admittance_set(
                                  bsh, distributed_ids[ i ], include_surplus_quadruples )
                                  .quadruples_map;
                    }
                    catch ( ... )
                    {
                        distributed_errors[ i ] = 1;
                    }
                }
            }
            {
                std::ostringstream errors_summary_stream;
                for ( std::size_t i = 0; i < distributed_errors.size(); i++ )
                {
                    if ( distributed_errors[ i ] != 0 )
                    {
                        errors_summary_stream << " " << i;
                    }
                }
                const std::string errors_summary = errors_summary_stream.str();
                if ( !errors_summary.empty() )
                {
                    throw std::runtime_error(
                        "Parallel processing failed because of exceptions in parts:" + errors_summary + "." );
                }
            }
        }
    };

    inline bool number_is_power_of_two( const unsigned long x ) { return ( ( x > 0 ) && ( ( x & ( x - 1 ) ) == 0 ) ); }
} // namespace voronota

#endif // APO_APP_COMMON_VORONOTA_HPP