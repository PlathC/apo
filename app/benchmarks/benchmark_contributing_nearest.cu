#include <fstream>

#include <apo/gpu/algorithm.cuh>
#include <common/samples.hpp>

namespace apo::gpu
{
    namespace impl
    {
        __global__ void getCellsContribution( ContextCellOriented    context,
                                              const uint32_t         k,
                                              const uint32_t * const knns,
                                              float * __restrict__ knownRadii,
                                              const uint32_t contributionNb,
                                              const uint32_t iterationNb,
                                              uint8_t * __restrict__ contribution,
                                              uint8_t * __restrict__ reachedSecurityRadius )
        {
            union TempStorage
            {
                WarpMergeSortNewEdgeT<MaxNewEdgeNb>::TempStorage newEdgeSort;
            };

            constexpr uint16_t     WarpPerBlock = BlockSize / WarpSize;
            __shared__ TempStorage tempStorages[ WarpPerBlock ];
            __shared__ Edge        edges[ TotalMaxEdgeNb * WarpPerBlock ];
            __shared__ Vertex      vertices[ TotalMaxVertexNb * WarpPerBlock ];

            const auto     block  = cg::this_thread_block();
            const auto     warp   = cg::tiled_partition<WarpSize>( block );
            const uint32_t warpId = threadIdx.x / WarpSize;

            const uint32_t start = blockIdx.x * WarpPerBlock;
            const uint32_t end   = apo::min( ( blockIdx.x + 1 ) * WarpPerBlock, context.siteNb );

            TempStorage & tempStorage = tempStorages[ warp.meta_group_rank() ];
            for ( uint32_t i = start + warpId; i < end; i += WarpPerBlock )
            {
                const uint8_t status = context.status[ i ];
                if ( status == CellStatus::FullyValidated || status == CellStatus::Buried )
                    continue;

                // Load edges and find neighbor list based on edges
                const uint32_t totalEdgeNb = context.edgeNb[ i ];
                CellEdges      cellEdges   = {
                    totalEdgeNb,
                    edges + TotalMaxEdgeNb * warp.meta_group_rank(),
                };

                for ( uint32_t e = warp.thread_rank(); e < totalEdgeNb; e += warpSize )
                {
                    const Edge edge      = context.edges[ i * TotalMaxEdgeNb + e ];
                    cellEdges.edges[ e ] = edge;
                }

                // Load vertices
                const uint32_t totalVertexNb = context.vertexNb[ i ];
                CellVertices   cellVertices  = {
                    totalVertexNb,
                    vertices + TotalMaxVertexNb * warp.meta_group_rank(),
                };

                for ( uint32_t v = warp.thread_rank(); v < totalVertexNb; v += warpSize )
                {
                    const Vertex vertex        = context.vertices[ i * TotalMaxVertexNb + v ];
                    cellVertices.vertices[ v ] = vertex;
                }

                // Synchronized for shared memory write
                warp.sync();

                const Real4 si          = context.sites[ i ];
                Real        knownRadius = -apo::Max; // Save reached distance to validate cell topology

                for ( uint32_t nId = 0; nId < k; nId++ )
                {
                    const uint32_t j = knns[ i * k + nId ];
                    if ( j == 0xffffffff )
                        break;

                    if ( context.status[ j ] == CellStatus::Buried )
                        continue;

                    const Real4 sj          = context.sites[ j ];
                    const bool  contributed = updateCell<TotalMaxVertexNb, TotalMaxEdgeNb, MaxNewEdgeNb>( //
                        warp,
                        context,
                        i,
                        si,
                        cellVertices,
                        cellEdges,
                        j,
                        sj,
                        tempStorage.newEdgeSort );

                    knownRadius = apo::max( knownRadius, apo::gpu::sphereDistance( si, sj ) );

                    if ( contributed && warp.thread_rank() == 0 )
                        contribution[ ( contributionNb * i ) + k * iterationNb + nId ] = 1;

                    // Check which part of the cell is validated from the security radius
                    const Real maxRadius      = cellRadius( warp, context, si, cellVertices, cellEdges );
                    const Real securityRadius = apo::Real( 2 ) * maxRadius;
                    const bool isCellOver     = apo::greaterThan( knownRadius, securityRadius );
                    if ( warp.thread_rank() == 0 && isCellOver )
                        reachedSecurityRadius[ ( contributionNb * i ) + k * iterationNb + nId ] = 1;

                    if ( warp.thread_rank() == 0 && isCellOver )
                        context.status[ i ] = CellStatus::FullyValidated;

                    if ( isCellOver )
                        break;
                }

                if ( threadIdx.x % warpSize == 0 )
                {
                    context.vertexNb[ i ] = cellVertices.size;
                    context.edgeNb[ i ]   = cellEdges.size;

                    knownRadii[ i ] = static_cast<float>( knownRadius );
                }

                for ( uint32_t e = warp.thread_rank(); e < cellEdges.size; e += warpSize )
                    context.edges[ i * TotalMaxEdgeNb + e ] = cellEdges.edges[ e ];

                for ( uint32_t v = warp.thread_rank(); v < cellVertices.size; v += warpSize )
                    context.vertices[ i * TotalMaxVertexNb + v ] = cellVertices.vertices[ v ];
            }
        }
    } // namespace impl

    template<uint32_t K = 16>
    std::tuple<std::vector<uint8_t>, std::vector<uint8_t>> compute( apo::ConstSpan<Real> m_sites, uint32_t iterationNb )
    {
        const uint32_t m_siteNb = m_sites.size / 4;

        Real m_minRadius = apo::Max;
        Real m_maxRadius = -apo::Max;
        Real m_minX      = apo::Max;
        Real m_minY      = apo::Max;
        Real m_minZ      = apo::Max;
        Real m_maxX      = -apo::Max;
        Real m_maxY      = -apo::Max;
        Real m_maxZ      = -apo::Max;
        for ( std::size_t v = 0; v < m_siteNb; v++ )
        {
            const Real x = m_sites[ v * 4 + 0 ];
            const Real y = m_sites[ v * 4 + 1 ];
            const Real z = m_sites[ v * 4 + 2 ];
            const Real r = m_sites[ v * 4 + 3 ];

            m_minX = apo::min( m_minX, x - r );
            m_minY = apo::min( m_minY, y - r );
            m_minZ = apo::min( m_minZ, z - r );

            m_maxX = apo::max( m_maxX, x + r );
            m_maxY = apo::max( m_maxY, y + r );
            m_maxZ = apo::max( m_maxZ, z + r );

            m_maxRadius = apo::max( m_maxRadius, r );
            m_minRadius = apo::min( m_minRadius, r );
        }

        ContextCellOriented context {};
        context.siteNb = m_siteNb;

        DeviceBuffer dSites = DeviceBuffer { ( m_siteNb + ArtificialSiteNb ) * sizeof( Real4 ), false };
        context.sites       = dSites.get<Real4>();
        context.minRadius   = m_minRadius;
        context.maxRadius   = m_maxRadius;

        // Storage for data structure
        DeviceBuffer dStatus = { m_siteNb * sizeof( uint8_t ), true };
        context.status       = dStatus.get<uint8_t>();

        DeviceBuffer dVertices   = DeviceBuffer { sizeof( Vertex ) * impl::TotalMaxVertexNb * m_siteNb, false };
        DeviceBuffer dVerticesNb = DeviceBuffer { sizeof( uint32_t ) * m_siteNb, true };
        context.vertexNb         = dVerticesNb.get<uint32_t>();
        context.vertices         = dVertices.get<Vertex>();

        DeviceBuffer dEdges   = DeviceBuffer { sizeof( Edge ) * impl::TotalMaxEdgeNb * m_siteNb, false };
        DeviceBuffer dEdgesNb = DeviceBuffer { sizeof( uint32_t ) * m_siteNb, true };
        context.edges         = dEdges.get<Edge>();
        context.edgeNb        = dEdgesNb.get<uint32_t>();

        const Real3 bbMin = Real3 { m_minX, m_minY, m_minZ };
        const Real3 bbMax = Real3 { m_maxX, m_maxY, m_maxZ };

        // Transfer sites to device and append artificial sites at the end
        mmemcpy<MemcpyType::HostToDevice>( dSites.get<Real>(), m_sites.ptr, m_sites.size );
        {
            const Real3 centroidCenter = ( bbMin + bbMax ) / apo::Real( 2 );
            const Real4 centroid       = { centroidCenter.x, centroidCenter.y, centroidCenter.z, Real( 0 ) };
            const Real  aabbRadius     = length(centroidCenter - Real3{m_maxX, m_maxY, m_maxZ});

            // Set a threshold to avoid numerical errors
            constexpr Real Threshold = Real( 1e-1 );
            const     Real radius    = aabbRadius * ( std::sqrt( Real( 3. ) ) + Threshold );

            const std::vector<Real4> artificialSites = {
                    centroid + Real4 { radius, Real( 0 ), Real( 0 ), context.minRadius },
                    centroid + Real4 { -radius, Real( 0 ), Real( 0 ), context.minRadius },
                    centroid + Real4 { Real( 0 ), radius, Real( 0 ), context.minRadius },
                    centroid + Real4 { Real( 0 ), -radius, Real( 0 ), context.minRadius },
                    centroid + Real4 { Real( 0 ), Real( 0 ), radius, context.minRadius },
                    centroid + Real4 { Real( 0 ), Real( 0 ), -radius, context.minRadius },
            };

            apo::joggle( { (Real *)artificialSites.data(), ArtificialSiteNb * 4 }, 1e-1 );

            mmemcpy<MemcpyType::HostToDevice>(
                    dSites.get<Real4>() + context.siteNb, artificialSites.data(), ArtificialSiteNb );
        }

        // Load artificial sites to constant memory
        constexpr std::size_t ArtificialVerticesDataCount = ArtificialVerticesNb * 4;
        constexpr std::size_t ArtificialEdgesDataCount    = ArtificialEdgesNb * 2;
        DeviceBuffer          artificialData
            = DeviceBuffer::Typed<uint32_t>( ArtificialVerticesDataCount + ArtificialEdgesDataCount );
        {
            const uint32_t artificialVerticesIndices[ ArtificialVerticesNb * 4 ] = {
                context.siteNb + 0, context.siteNb + 2, context.siteNb + 4, 0, //
                context.siteNb + 0, context.siteNb + 2, context.siteNb + 5, 0, //
                context.siteNb + 0, context.siteNb + 3, context.siteNb + 4, 0, //
                context.siteNb + 0, context.siteNb + 3, context.siteNb + 5, 0, //
                context.siteNb + 1, context.siteNb + 2, context.siteNb + 4, 0, //
                context.siteNb + 1, context.siteNb + 2, context.siteNb + 5, 0, //
                context.siteNb + 1, context.siteNb + 3, context.siteNb + 4, 0, //
                context.siteNb + 1, context.siteNb + 3, context.siteNb + 5, 0, //
            };

            apo::gpu::mmemcpy<MemcpyType::HostToDevice>(
                artificialData.get<uint32_t>(), artificialVerticesIndices, ArtificialVerticesDataCount );

            const uint32_t artificialEdgesIndices[ ArtificialEdgesNb * 2 ] = {
                context.siteNb + 0, context.siteNb + 2, //
                context.siteNb + 2, context.siteNb + 4, //
                context.siteNb + 1, context.siteNb + 2, //
                context.siteNb + 2, context.siteNb + 5, //
                context.siteNb + 1, context.siteNb + 4, //
                context.siteNb + 1, context.siteNb + 5, //
                context.siteNb + 0, context.siteNb + 5, //
                context.siteNb + 0, context.siteNb + 4, //
                context.siteNb + 1, context.siteNb + 3, //
                context.siteNb + 3, context.siteNb + 5, //
                context.siteNb + 0, context.siteNb + 3, //
                context.siteNb + 3, context.siteNb + 4, //
            };

            apo::gpu::mmemcpy<MemcpyType::HostToDevice>( //
                artificialData.get<uint32_t>() + ArtificialVerticesDataCount,
                artificialEdgesIndices,
                ArtificialEdgesDataCount );
        }

        // BVH construction
        LBVH  bvh  = LBVH {};
        Aabbf aabb = { make_float3( m_minX, m_minY, m_minZ ), make_float3( m_maxX, m_maxY, m_maxZ ) };
        bvh.build( context.siteNb, aabb, context.sites );

        // Actual computation
        // Initialize data structure
        auto [ gridDim, blockDim ] = KernelConfig::From( context.siteNb, impl::KnnBlockSize );
        initializeDataStructure<impl::TotalMaxVertexNb, impl::TotalMaxEdgeNb><<<gridDim, blockDim>>>(
            context, artificialData.get<uint32_t>(), artificialData.get<uint32_t>() + ArtificialVerticesDataCount );
        apo::gpu::cudaCheck( "initializeDataStructure" );

        DeviceBuffer dKnns = DeviceBuffer( m_siteNb * K * ( sizeof( uint32_t ) + sizeof( Real ) ), false );
        cudaCheck( cudaMemset( dKnns.get<uint32_t>(), 0xff, sizeof( uint32_t ) * m_siteNb * K ) );

        DeviceBuffer validatedEdgesNb    = DeviceBuffer::Typed<uint32_t>( context.siteNb, true );
        DeviceBuffer validatedVerticesNb = DeviceBuffer::Typed<uint32_t>( context.siteNb, true );

        std::vector<float> initRadii  = std::vector<float>( context.siteNb, std::numeric_limits<float>::lowest() );
        DeviceBuffer       knownRadii = DeviceBuffer::Typed<float>( context.siteNb, false );
        mmemcpy<MemcpyType::HostToDevice>( knownRadii.get<float>(), initRadii.data(), context.siteNb );

        DeviceBuffer maxEdgeNb      = DeviceBuffer::Typed<uint32_t>( 1, true );
        DeviceBuffer maxVertexNb    = DeviceBuffer::Typed<uint32_t>( 1, true );
        DeviceBuffer finishedCellNb = DeviceBuffer::Typed<uint32_t>( 1, true );

        uint32_t     knnValidationStep      = ( iterationNb / K ) + 1;
        DeviceBuffer dContribution          = DeviceBuffer::Typed<uint8_t>( m_siteNb * knnValidationStep * K, true );
        DeviceBuffer dReachedSecurityRadius = DeviceBuffer::Typed<uint8_t>( m_siteNb * knnValidationStep * K, true );
        for ( uint32_t s = 0; s < knnValidationStep; s++ )
        {
            impl::getKnns<K><<<gridDim, blockDim>>>( s, context, bvh.getDeviceView(), dKnns.get<uint32_t>() );
            apo::gpu::cudaCheck( "getKnns" );

            auto [ cellGridDim, cellBlockDim ] = KernelConfig::From( context.siteNb * WarpSize, impl::BlockSize );
            impl::getCellsContribution<<<cellGridDim, cellBlockDim>>>( context,
                                                                       K,
                                                                       dKnns.get<uint32_t>(),
                                                                       knownRadii.get<float>(),
                                                                       knnValidationStep * K,
                                                                       s,
                                                                       dContribution.get<uint8_t>(),
                                                                       dReachedSecurityRadius.get<uint8_t>() );
            apo::gpu::cudaCheck( "getCells" );
        }

        return std::make_tuple( dContribution.toHost<uint8_t>(), dReachedSecurityRadius.toHost<uint8_t>() );
    }
} // namespace apo::gpu

int main( int, char ** )
{
    constexpr uint32_t ConsideredNb = 1024;

    {
        const apo::Path        path   = "samples/1AON.mmtf";
        std::vector<apo::Real> sites  = apo::loadProtein( path );
        const uint32_t         siteNb = sites.size() / 4;
        const std::string      pdb    = std::filesystem::path( path ).stem().string();

        apo::logger::info( "Perform nearest benchmarks on {} with {} sites", pdb, siteNb );

        apo::joggle( sites, 1e-3 );

        const auto [ proteinContributions, proteinReachedSecurityRadius ] = apo::gpu::compute( sites, ConsideredNb );
        const uint32_t stride                                             = proteinContributions.size() / siteNb;

        std::vector<char> out {};
        fmt::format_to( std::back_inserter( out ), "SiteId;Iteration;Contributed;ReachedSecurityRadius\n" );

        for ( std::size_t i = 0; i < siteNb; i++ )
            for ( std::size_t j = 0; j < ConsideredNb; j++ )
                fmt::format_to( std::back_inserter( out ),
                                "{};{};{};{}\n",
                                i,
                                j,
                                proteinContributions[ ( stride * i ) + j ],
                                proteinReachedSecurityRadius[ ( stride * i ) + j ] );

        auto outputCsv      = std::ofstream { fmt::format( "./nearest-study-proteins-{}.csv", pdb ) };
        auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
        std::copy( out.begin(), out.end(), outputIterator );
    }

    {
        struct CloudConfiguration
        {
            uint32_t  count;
            apo::Real spreading;
            apo::Real radiiFactor;
            apo::Real radiiStart;
        };

        CloudConfiguration     configuration { 100000, 500., 2., 1. };
        std::vector<apo::Real> sites = apo::getUniform(
            configuration.count, configuration.spreading, configuration.radiiFactor, configuration.radiiStart );

        apo::logger::info(
            "Perform nearest benchmark for configuration of sites: {}, spreading: {}, radiiFactor: {}, radiiStart: {}",
            configuration.count,
            configuration.spreading,
            configuration.radiiFactor,
            configuration.radiiStart );

        apo::joggle( sites, 1e-3 );

        const auto [ cloudContributions, cloudReachedSecurityRadius ] = apo::gpu::compute( sites, ConsideredNb );
        const uint32_t stride                                         = cloudContributions.size() / configuration.count;

        std::vector<char> out {};
        fmt::format_to( std::back_inserter( out ), "SiteId;Iteration;Contributed;ReachedSecurityRadius\n" );

        for ( std::size_t i = 0; i < configuration.count; i++ )
            for ( std::size_t j = 0; j < ConsideredNb; j++ )
                fmt::format_to( std::back_inserter( out ),
                                "{};{};{};{}\n",
                                i,
                                j,
                                cloudContributions[ ( stride * i ) + j ],
                                cloudReachedSecurityRadius[ ( stride * i ) + j ] );

        auto outputCsv      = std::ofstream { fmt::format( "./nearest-study-clouds-{}-{}-{}-{}.csv",
                                                      configuration.count,
                                                      configuration.spreading,
                                                      configuration.radiiFactor,
                                                      configuration.radiiStart ) };
        auto outputIterator = std::ostream_iterator<char> { outputCsv, "" };
        std::copy( out.begin(), out.end(), outputIterator );
    }

    return EXIT_SUCCESS;
}