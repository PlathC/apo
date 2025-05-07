#include "common/samples.hpp"

#include <fstream>
#include <sstream>

#include <chemfiles.hpp>

namespace apo
{
    std::vector<Real> loadProtein( const apo::Path & path )
    {
        static bool configured = false;
        if ( !configured )
        {
            configured = true;
#ifndef NDEBUG
            chemfiles::warning_callback_t callback
                = []( const std::string & p_log ) { apo::logger::debug( "{}\n", p_log ); };
#else
            chemfiles::warning_callback_t callback = []( const std::string & p_log ) {};
#endif
            chemfiles::set_warning_callback( callback );
        }

        chemfiles::Trajectory trajectory { path.string() };
        if ( trajectory.nsteps() == 0 )
            throw std::runtime_error( "Trajectory is empty" );

        chemfiles::Frame                        frame    = trajectory.read();
        const chemfiles::Topology &             topology = frame.topology();
        const std::vector<chemfiles::Residue> & residues = topology.residues();
        const std::vector<chemfiles::Bond> &    bonds    = topology.bonds();

        if ( frame.size() != topology.size() )
            throw std::runtime_error( "Data count missmatch" );

        // Set molecule properties.
        std::string name;
        if ( frame.get( "name" ) )
            name = frame.get( "name" )->as_string();

        std::vector<apo::Real> molecule {};
        for ( const chemfiles::Residue & residue : residues )
        {
            for ( const std::size_t atomId : residue )
            {
                const chemfiles::Atom & atom = topology[ atomId ];

                const chemfiles::span<chemfiles::Vector3D> & positions = frame.positions();
                const chemfiles::Vector3D &                  position  = positions[ atomId ];

                const auto vdWRadius = atom.vdw_radius();
                molecule.emplace_back( apo::Real( position[ 0 ] ) );
                molecule.emplace_back( apo::Real( position[ 1 ] ) );
                molecule.emplace_back( apo::Real( position[ 2 ] ) );
                molecule.emplace_back( vdWRadius ? apo::Real( *vdWRadius ) : apo::Real( 1 ) );
            }
        }

        return molecule;
    }

    // Reference: https://www.shadertoy.com/view/XlGcRh
    static Real pcg( uint32_t v )
    {
        uint32_t state = v * 747796405u + 2891336453u;
        uint32_t word  = ( ( state >> ( ( state >> 28u ) + 4u ) ) ^ state ) * 277803737u;
        return apo::Real( ( word >> 22u ) ^ word ) / apo::Real( std::numeric_limits<uint32_t>::max() );
    }

    std::vector<Real> getUniform( uint32_t count, Real spreading, Real radiiFactor, Real radiiStart )
    {
        std::vector<apo::Real> data {};
        data.reserve( count );

        for ( uint32_t i = 0; i < count; i++ )
        {
            data.emplace_back( pcg( i * 4 + 0 ) * spreading );
            data.emplace_back( pcg( i * 4 + 1 ) * spreading );
            data.emplace_back( pcg( i * 4 + 2 ) * spreading );
            data.emplace_back( pcg( i * 4 + 3 ) * radiiFactor + radiiStart );
        }

        return data;
    }

    std::vector<apo::Real> parseFromDataset( const apo::Path & path )
    {
        // Based on https://stackoverflow.com/a/70077787
        std::ifstream is = std::ifstream { path };
        if ( !is.is_open() )
            throw std::runtime_error( "Can't find file " + path.string() );

        // First row contains siteNb
        std::string line;
        std::getline( is, line );
        const uint32_t siteNb = std::stoul( line );

        // Parse sites
        std::vector<apo::Real> sites {};
        sites.reserve( siteNb * 4 );
        while ( std::getline( is, line ) )
        {
            std::replace( line.begin(), line.end(), ' ', '\t' );

            std::stringstream ss = std::stringstream( line );

            std::string              token;
            std::vector<std::string> temp;
            while ( getline( ss, token, '\t' ) )
                if ( !token.empty() )
                    temp.push_back( token );

            const apo::Real x = std::stof( temp[ 1 ] );
            const apo::Real y = std::stof( temp[ 2 ] );
            const apo::Real z = std::stof( temp[ 3 ] );
            const apo::Real r = std::stof( temp[ 4 ] );

            sites.emplace_back( x );
            sites.emplace_back( y );
            sites.emplace_back( z );
            sites.emplace_back( r );
        }
        is.close();

        return sites;
    }
} // namespace apo