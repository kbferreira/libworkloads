#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include "stencil.h"
#include <mpi.h>
#include "checks.h"
#include "rdtsc.h"
#include "timing.h"
#include <math.h>
#include "workloads.h"

static int npx = -1, npy = -1, npz = -1;
static int *neighbors = NULL;
static int num_neighbors = 0;
static int duration = 0;
static int p2p_size  =  0, coll_size = 0, p2p_size_local = 0,
           p2p_size_min, p2p_size_max, p2p_size_sum;
static double p2p_size_avg;

static int opt_fd = -1;

#define FNAME_MAX ( 100 )

#define KERNEL_NAME     "Stencil"
#define PROBLEM_DESC    "3D, 27pt exchange"

static double calibration_start, calibration_end, calibration_avg, 
              calibration_avg_local, calibration_min, calibration_max;
static double collection_start, collection_end, collection_min,
              collection_avg, collection_avg_local = 0.0, collection_max;

static int
do_stats( MPI_Comm this_comm ){
        int rank, size;

        MPI_CHECK( PMPI_Comm_rank( this_comm, &rank ) );
        MPI_CHECK( PMPI_Comm_size( this_comm, &size ) );

        calibration_avg = calibration_avg_local = calibration_end - calibration_start;
        MPI_CHECK( PMPI_Allreduce( &calibration_avg_local, &calibration_avg, 1, 
                                  MPI_DOUBLE, MPI_SUM, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &calibration_avg_local, &calibration_min, 1,
                                  MPI_DOUBLE, MPI_MIN, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &calibration_avg_local, &calibration_max, 1,
                                  MPI_DOUBLE, MPI_MAX, this_comm ) );
        calibration_avg = calibration_avg / size;

        collection_avg = collection_avg_local;
        MPI_CHECK( PMPI_Allreduce( &collection_avg_local, &collection_min, 1,
                                  MPI_DOUBLE, MPI_MIN, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &collection_avg_local, &collection_max, 1,
                                  MPI_DOUBLE, MPI_MAX, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &collection_avg_local, &collection_avg, 1,
                                  MPI_DOUBLE, MPI_SUM, this_comm ) );
        collection_avg = collection_avg / size;

        MPI_CHECK( PMPI_Allreduce( &p2p_size_local, &p2p_size_min, 1,
                                  MPI_INT, MPI_MIN, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &p2p_size_local, &p2p_size_max, 1,
                                  MPI_INT, MPI_MAX, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &p2p_size_local, &p2p_size_sum, 1,
                                  MPI_INT, MPI_SUM, this_comm ) );
        p2p_size_avg = ( double )p2p_size_sum / ( double )size;

        _workloads_stats_generated = 1;

        return 0;

}

void
stencil_summary( FILE *fp )
{
        assert( fp != NULL );

        if( !_workloads_stats_generated )
                do_stats( workload_comm );

        fprintf( fp, "\nWorkload: %s.\nProblem description: %s\n", KERNEL_NAME,
                     PROBLEM_DESC );
        fprintf( fp, "Process grid (x,y,z): (%d x %d x %d)\n",
                        npx, npy, npz );
        fprintf( fp, "Work iteration length:\t\t\t %10d usecs\n", duration );
        fprintf( fp, "Agreed upon P2P size:\t\t\t %10d\nAgreed upon Collective size:\t\t %10d\n",
                        p2p_size * ( int )sizeof( int ), 
                        coll_size * ( int )sizeof( int ) );
        fprintf( fp, "Workload distribution:\t [ %d, %.0lf, %d ]\n",
                        p2p_size_min * ( int )sizeof( int ),
                        p2p_size_avg * ( double )sizeof( int ),
                        p2p_size_max * ( int )sizeof( int ) );
        fprintf( fp, "Calibration Times:\t [ %.2lf, %.2lf, %.2lf ] secs\n",
                        calibration_min, calibration_avg, calibration_max );
        fprintf( fp, "Collection Times:\t [ %.2lf, %.2lf, %.2lf ] secs\n",
                        collection_min, collection_avg, collection_max );

        return;
}
int
stencil_write( FILE *fp, void *v, void *p )
{
        uint64_t start;
        dlist_t  *d = NULL;

        assert( v != NULL );

        d = ( dlist_t *)v;

        assert( benchmark_duration != 0 );

        /*
         * We use ticks_per_second parameter, so make
         * sure it has been properly set -kbf
         */
        assert( ticks_per_second != 0 );

        if( !_workloads_stats_generated )
                do_stats( workload_comm );

        if( fp == NULL )
                fp = stdout;

        fprintf( fp, "# %s data file. Problem description: %s\n", KERNEL_NAME,
                     PROBLEM_DESC );
        fprintf( fp, "# %s data run [ %.6lf %.6lf ]\n", KERNEL_NAME,
                        d->run_start - _workloads_epoch,
                        d->run_end - _workloads_epoch );
        fprintf( fp, "# Process grid (x,y,z): (%d x %d x %d)\n",
                        npx, npy, npz );
        fprintf( fp, "# Number of neighbors: %d\n", num_neighbors );
        fprintf( fp, "# Agreed p2p size: %d, local p2p size: %d, coll size: %d\n",
                        p2p_size, p2p_size_local, coll_size );
        fprintf( fp, "# calibrated ticks per second: %" PRIu64 "\n", ticks_per_second );
        fprintf( fp, "# Requested no-interference kernel duration: %d secs\n", 
                        benchmark_duration );
        fprintf( fp, "# Requested compute iteration length: %d usecs\n",
                      duration );
        fprintf( fp, "# Local Calibration duration: %.2lf secs\n", 
                        calibration_avg_local );
        fprintf( fp, "# Local stencil collection time: %.2lf\n",
                        collection_avg_local );
        fprintf( fp, "# index\t duration (nsec.)\t wait (nsec.)\n" );

        for( int i = 0; i < d->n; i++ ){
                fprintf( fp, "%d\t %" PRIu64 "\t %" PRIu64 "\n",
                         d->list[ i ].c.indx, 
                         ( uint64_t )( ceil( 1.0e9 * 
                                     ( ( double )( d->list[ i ].c.delta ) /
                                     ( double )( ticks_per_second ) ) ) ),
                         ( uint64_t )( ceil( 1.0e9 * 
                                     ( ( double )( d->list[ i ].c.wait ) /
                                     ( double )( ticks_per_second ) ) ) ) );
        }
        return d->n;
}

static int
setup_neighbors_3d27pt( int rank, int size )
{
        int cart_rank;
        const int ndirs = 26;
        const int dim_wrap = 1; /* do not wrap around */
        MPI_Comm grid;
        int dims[ 3 ], periods[ 3 ], coords[ 3 ];

        assert( ( rank >= 0 ) && ( size >= 1 ) );

        neighbors = malloc( ndirs * sizeof( int ) );
        assert( neighbors );

        num_neighbors = 0;

        for( int i = 0 ; i < ndirs ; i++ ){
                neighbors[ i ] = -1;
        }

        dims[ 0 ] = npx;
        dims[ 1 ] = npy;
        dims[ 2 ] = npz;

        periods[ 0 ] = periods[ 1 ] = periods[ 2 ] = dim_wrap;

        MPI_CHECK( PMPI_Cart_create( workload_comm, 3, dims, 
                                    periods, 0, &grid ) );
        MPI_CHECK( PMPI_Comm_rank( grid, &cart_rank ) );

        MPI_CHECK( PMPI_Cart_coords( grid, cart_rank, 3, coords ) );

        for( int i = -1 ; i <= 1 ; i++ ){
                for( int j = -1 ; j <= 1; j++ ){
                        for( int k = -1; k <= 1 ; k++ ){
                               /*
                                * Skip me
                                */
                               if( ( i == 0 ) && ( j == 0 ) && ( k == 0 ) ){
                                      continue;
                               }

                               dims[ 0 ] = coords[ 0 ] + i;
                               dims[ 1 ] = coords[ 1 ] + j;
                               dims[ 2 ] = coords[ 2 ] + k;

#if 0
                               /* 
                                * this is needed if wrap dimensions -kbf
                                */
                               if( ( ( dims[ 0 ] >= 0 ) && ( dims[ 0 ] < npx ) ) &&
                                   ( ( dims[ 1 ] >= 0 ) && ( dims[ 1 ] < npy ) ) &&
                                   ( ( dims[ 2 ] >= 0 ) && ( dims[ 2 ] < npz ) ) )
                               {
#endif /* 0 */
                              MPI_CHECK( PMPI_Cart_rank( grid, dims, &cart_rank ) );
                              assert( cart_rank != MPI_PROC_NULL );

                              neighbors[ num_neighbors++ ] = cart_rank;

#if 0
                              /*
                               * See note above -kbf
                               */
                               }
#endif /* 0 */
                        }
                }
        }

        MPI_CHECK( PMPI_Comm_free( &grid ) );

        return 0;
}

static int
compute_it( int *p2p_rbuf, int *p2p_data, int p_size,
            int *coll_rbuf, int *coll_data, int c_size )
{
        int result;
        
        for( int i = 0 ; i < ( p_size * num_neighbors ) ; i++ ){
                        p2p_data[ i ]  = ( p2p_data[ i ]  + p2p_rbuf[ i ] ) / 2;
                        result +=  p2p_data[i ];
        }

        for( int i = 0 ; i < c_size ; i++ ){
                result += coll_data[ i ];
        }

        return result;
}

static uint64_t
calibrate_compute( int usecs )
{
        double desired_ticks;
        uint64_t min, lmin;
        int array_count = 1024 * 1024;
        int *rbuf = NULL, *data = NULL, *c_rbuf = NULL, *c_data = NULL;
        int result = 0;
        char tmpfname[ FNAME_MAX ];
        int rank;

        MPI_CHECK( PMPI_Comm_rank( workload_comm, &rank ) );

        INIT_STENCIL;

        min = 0;

        if( ticks_per_second == 0 ){
                ticks_per_second = get_ticks_per_second( 200 );
        }

        assert( ticks_per_second != 0 );
        assert( num_neighbors != 0 );

        rbuf = malloc( sizeof( int ) * array_count * num_neighbors );
        assert( rbuf != NULL );

        data = malloc( sizeof( int ) * array_count * num_neighbors );
        assert( data != NULL );

        c_rbuf = malloc( sizeof( int ) * 1 );
        assert( c_rbuf != NULL );

        c_data = malloc( sizeof( int ) * 1 );
        assert( c_data );

        desired_ticks = ( ticks_per_second / 1e6 ) * usecs;

        while( ( min == 0 ) ||
               ( ( fabs( desired_ticks - min ) / desired_ticks ) > 0.005 ) )
        {
                uint64_t t1, t2;
                min = 0;

                for( int i = 0 ; i < 50 ; i++ ){
                        t1 = rdtsc();
                        result += compute_it( rbuf, data, array_count,
                                                      c_rbuf, c_data, 1 );
                        t2 = rdtsc();

                        lmin = t2 - t1;

                        if( ( min == 0 ) || ( min > lmin ) ){
                                min = lmin;
                        }
                }

                array_count = ( uint64_t )( ceil( ( desired_ticks / 
                                        ( double )min ) * ( double )array_count ) );

                rbuf = realloc( rbuf, sizeof( int ) * array_count * num_neighbors );
                assert( rbuf != NULL );

                data = realloc( data, sizeof( int ) * array_count * num_neighbors );
                assert( data != NULL );
        }

        FINALIZE_STENCIL( result );

        free( rbuf );
        free( data );
        free( c_rbuf );
        free( c_data );

        return array_count;
}

static int
queue_communication_3d27pt( int *p2p_rbuf, int *p2p_data, int p_size, 
                            int *coll_rbuf, int *coll_data, int c_size,
                            MPI_Request *reqs, int rmax, int *rused )
{
        const int p2p_tag = 1234;
        int result = 0;

        assert( neighbors != NULL );
        assert( ( rmax > 0 ) && ( reqs != NULL ) );
        assert( rused != NULL );
        assert( ( p_size > 0 ) && ( p2p_rbuf != NULL ) && ( p2p_data != NULL ) );
        assert( ( c_size > 0 ) && ( coll_rbuf != NULL ) && ( coll_data != NULL ) );

        *rused = 0;

        assert( rmax > *rused );

        for( int i = 0; i < num_neighbors ; i++ ){
                if( neighbors[ i ] != -1 ){
                        MPI_CHECK( PMPI_Irecv( p2p_rbuf + ( i * p_size ), 
                                              p_size, MPI_INT,
                                              neighbors[ i ], p2p_tag,
                                              workload_comm,
                                              &reqs[ (*rused)++ ] ) );
                        assert( rmax > *rused );
                }

                if( neighbors[ i ] != -1 ){
                        MPI_CHECK( PMPI_Isend( p2p_data + ( i * p_size ),
                                              p_size, MPI_INT,
                                              neighbors[ i ], p2p_tag,
                                              workload_comm,
                                              &reqs[ ( *rused )++ ] ) );
                        assert( rmax > *rused );
                }
        }

        MPI_CHECK( PMPI_Iallreduce( coll_data, coll_rbuf, c_size, MPI_INT,
                                   MPI_SUM, workload_comm, &reqs[ ( *rused )++ ] ) );

        result += compute_it( p2p_rbuf, p2p_data, p_size,
                              coll_rbuf, coll_data, c_size );

        return result;
}

static int
wait_for_communication_3d27pt( MPI_Request *reqs, int size )
{
        assert( ( size > 0 ) && ( reqs != NULL ) );

        MPI_CHECK( PMPI_Waitall( size, reqs, MPI_STATUSES_IGNORE ) );

        return 0;
}

void
stencil_teardown( void *p )
{
        int rank, np;
        dlist_t *d = NULL;

        assert( p != NULL );

        if( _workloads_initialized != 1 ){
                fprintf( stderr, "Calling teardown() when !initialized\n" );
                return;
        }

        d = ( dlist_t * )p;

        if( d->list != NULL ){
                free( d->list );
                d->max = d->n = 0;
        }

        MPI_CHECK( PMPI_Comm_free( &workload_comm ) );

        _workloads_initialized = 0;

        return;
}

static int
do_run( dlist_t *d, void *p2p_rbuf, void *p2p_data, int p_size,
        void *coll_rbuf, void *coll_data, int c_size, MPI_Request *reqs, int rmax,
        int *rused )
{
        double the_start, the_end;
        uint64_t start, end;
        long sample = 0, nruns;
        int result = 0, rank;
        char tmpfname[ FNAME_MAX ];

        MPI_CHECK( PMPI_Comm_rank( workload_comm, &rank ) );

        INIT_STENCIL;

        assert( benchmark_duration != 0 );

        nruns = d->max;
        MPI_CHECK( PMPI_Allreduce( MPI_IN_PLACE, &nruns, 1, MPI_INT, MPI_MIN,
                                  workload_comm ) );

        if( workload_verbose )
                fprintf( stderr, "# nruns: %ld\n",  nruns );

        /*
         * FIXME: This timing stuff should be a bit$
         *        better abstracted and not directly use MPI$
         *        as that might notbe suffucent resilution$
         */
        the_start = PMPI_Wtime();
        the_end = the_start + benchmark_duration;

        while( sample < nruns ){
                start = rdtsc();
                
                result += queue_communication_3d27pt( p2p_rbuf, p2p_data, p_size,
                                           coll_rbuf, coll_data, c_size,
                                           reqs, rmax, rused );
                end = rdtsc();
        
                d->list[ sample ].c.delta = end - start;
                d->list[ sample ].c.indx = sample;

                start = rdtsc();
                wait_for_communication_3d27pt( reqs, *rused );
                end = rdtsc();

                d->list[ sample ].c.wait = end - start;

                sample++;

        }

        d->n = sample;

        FINALIZE_STENCIL( result );

        return sample;
}

int
stencil_setup( void *priv, void *p )
{
        int p2p_local_size;
        int rank, np;
        dlist_t *d = NULL;
        long dpoints = 0;

        stencil_params_t *s;

        assert( priv != NULL );
        assert( p != NULL );

        if( _workloads_initialized != 0 ){
                fprintf( stderr, "Calling setup() when already initialized\n" );
                return -1;
        }

        d = ( dlist_t *)priv;
        s = ( stencil_params_t *)p;

        benchmark_duration = d->duration;
        workload_verbose = d->verbose;

        MPI_CHECK( PMPI_Comm_dup( s->comm, &workload_comm ) );
        MPI_CHECK( PMPI_Comm_rank( workload_comm, &rank ) );
        MPI_CHECK( PMPI_Comm_size( workload_comm, &np ) );

        npx = s->npx;
        npy = s->npy;
        npz = s->npz;
        duration = s->usecs;

        dpoints = ( long )( ceil( benchmark_duration / ( duration * 1.0e-6 ) ) );
        d->list = malloc( sizeof( datapoint_t ) * dpoints );
        assert( d->list != NULL );

        d->n = 0;
        d->max = dpoints;

        if( ( workload_verbose ) && ( rank == 0 ) ){
                fprintf( stderr, "# Process grid (x,y,z): [%d x %d x %d]\n# np: %d\n",
                                 npx, npy, npz, np );
        }

        assert( np == ( npx * npy * npz ) );

        /*
         * FIXME: Direct calls to MPI_Wtime() seems like poor design -kbf
         */
        calibration_start = PMPI_Wtime();

        setup_neighbors_3d27pt( rank, np );

        /*
         * Calibrate the compute portion
         */
        if( ( workload_verbose ) && ( rank == 0 ) ){
                fprintf( stdout, "Calibrating stencil computation\n" );
        }

        p2p_size_local = calibrate_compute( duration );
        coll_size = 1;

        if( workload_verbose ){
                fprintf( stderr, "# P2P size: %d, COLL size: %d\n",
                                 p2p_size_local, coll_size );
        }

        MPI_CHECK( PMPI_Allreduce( &p2p_size_local, &p2p_size, 1, MPI_INT, 
                                  MPI_MIN, workload_comm ) );

        if( ( workload_verbose ) && rank == 0 ){
                fprintf( stderr, "# Agred upon P2P size: %d, COLL size: %d\n",
                                 p2p_size, coll_size );
        }
        calibration_end = PMPI_Wtime();

        _workloads_stats_generated = 0;
        _workloads_initialized = 1;

        return 0;

}

int
stencil_run( void *v, void *private)
{
        dlist_t *d = NULL;
        double the_start, the_end;
        stencil_params_t *s = NULL;
        int rank, np;
        int *p2p_rbuf = NULL, *p2p_data = NULL;
        int *coll_rbuf = NULL, *coll_data = NULL;
        int random_result = 0;
        MPI_Request *reqs = NULL;
        int reqs_used;
        uint64_t min;
        char tmpfname[ FNAME_MAX ];

        if( _workloads_initialized != 1 ){
                fprintf( stderr, "Calling run() when !initialized\n" );
                return -1;
        }

        assert( v != NULL );

        d = ( dlist_t * )v;

        p2p_rbuf = malloc( sizeof( int ) * ( p2p_size * num_neighbors ) );
        assert( p2p_rbuf != NULL );

        p2p_data = malloc( sizeof( int ) * ( p2p_size * num_neighbors ) );
        assert( p2p_data != NULL );

        coll_rbuf = malloc( sizeof( int ) * coll_size );
        assert( coll_rbuf != NULL );

        coll_data = malloc( sizeof( int ) * coll_size );
        assert( coll_data != NULL );

        reqs = malloc( sizeof( MPI_Request ) * ( ( num_neighbors * 2 ) + 1 ) );

        for( int i = 0 ; i < p2p_size ; i++ ){
                p2p_data[ i ] = rand();
        }

        for( int i = 0 ; i < coll_size ; i++ ){
                coll_data[ i ] = rand();
        }

        if( ( workload_verbose ) && ( rank == 0 ) ){
                fprintf( stdout, "Starting Stencil pattern collection\n" );
        }

        /*
         * FIXME: Direct calls to MPI_Wtime() seem like poor design -kbf
         */
        d->run_start = collection_start = PMPI_Wtime();

        do_run( d, p2p_rbuf, p2p_data, p2p_size,
                coll_rbuf, coll_data, coll_size,
                reqs, ( num_neighbors * 2 ) + 1, &reqs_used );

        free( p2p_rbuf );
        free( p2p_data );
        free( coll_rbuf );
        free( coll_data );
        free( reqs );

        d->run_end = collection_end = PMPI_Wtime();

        collection_avg_local += ( collection_end - collection_start );

        _workloads_stats_generated = 0;

        return d->n;
}
