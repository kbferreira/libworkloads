#include "fwq.h"
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include "rdtsc.h"
#include "checks.h"
#include "timing.h"
#include "workloads.h"

static int usecs;
static uint64_t workload_runs;

static double calibration_start, calibration_end, calibration_avg,
              calibration_avg_local, calibration_min, calibration_max;;
static double collection_start, collection_end, collection_min,
              collection_avg, collection_avg_local = 0.0, collection_max;
static double workload_min, workload_max, workload_avg;

#define KERNEL_NAME "Fixed Work Quantum"
#define PROBLEM_DESC "Cache-friendly workload from netgauge"

static int
do_stats( MPI_Comm this_comm ){
        int rank, size;


        MPI_CHECK( PMPI_Comm_rank( this_comm, &rank ) );
        MPI_CHECK( PMPI_Comm_size( this_comm, &size ) );

        calibration_avg = calibration_avg_local = calibration_end - calibration_start;
        MPI_CHECK( PMPI_Allreduce( &calibration_avg_local, &calibration_min, 1,
                                  MPI_DOUBLE, MPI_MIN, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &calibration_avg_local, &calibration_max, 1,
                                  MPI_DOUBLE, MPI_MAX, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &calibration_avg_local, &calibration_avg, 1,
                                  MPI_DOUBLE, MPI_SUM, this_comm ) );
        calibration_avg = calibration_avg / size;

        collection_avg = collection_avg_local;
        MPI_CHECK( PMPI_Allreduce( &collection_avg_local, &collection_min, 1,
                                  MPI_DOUBLE, MPI_MIN, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &collection_avg_local, &collection_max, 1,
                                  MPI_DOUBLE, MPI_MAX, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &collection_avg_local, &collection_avg, 1,
                                  MPI_DOUBLE, MPI_SUM, this_comm ) );
        collection_avg = collection_avg / size;

        workload_avg = ( double )workload_runs;
        MPI_CHECK( PMPI_Allreduce( &workload_avg, &workload_min, 1, MPI_DOUBLE,
                                  MPI_MIN, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &workload_avg, &workload_max, 1, MPI_DOUBLE,
                                  MPI_MAX, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( MPI_IN_PLACE, &workload_avg, 1, MPI_DOUBLE,
                                  MPI_SUM, this_comm ) );
        workload_avg = workload_avg / size;

        _workloads_stats_generated = 1;

        return 0;
}

void
fwq_summary( FILE *fp )
{
        assert( fp != NULL );

        if( !_workloads_stats_generated )
                do_stats( workload_comm );

        fprintf( fp, "\nWorkload: %s.\nProblem description: %s\n", KERNEL_NAME,
                     PROBLEM_DESC );
        fprintf( fp, "Work iteration length:\t\t\t %10d usecs\n", usecs );
        fprintf( fp, "Calibration Times:\t [ %.2lf, %.2lf, %.2lf ] secs\n",
                        calibration_min, calibration_avg, calibration_max );
        fprintf( fp, "Collection Times:\t [ %.2lf, %.2lf, %.2lf ] secs\n",
                        collection_min, collection_avg, collection_max );
        fprintf( fp, "Workload distribution:\t [ %.2lf, %.2lf, %.2lf ]\n",
                        workload_min, workload_avg, workload_max );

        return;
}

int
fwq_write( FILE *fp, void  *v, void *p )
{
        dlist_t *d = NULL;
        int rank, size;

        assert( v != NULL );

        d = ( dlist_t *)v;

        assert( benchmark_duration != 0 );

        /*
         * We use ticks_per_second parameter, so make
         * sure it as been properly set -kbf
         */
        assert( ticks_per_second != 0 );
        
        if( !_workloads_stats_generated )
                do_stats( workload_comm );

        fprintf( fp, "# %s data file. Problem description: %s\n", KERNEL_NAME,
                     PROBLEM_DESC );
        fprintf( fp, "# %s data run [ %.6lf %.6lf ]\n", KERNEL_NAME,
                        d->run_start - _workloads_epoch,
                        d->run_end - _workloads_epoch );
        fprintf( fp, "# Workload count: %" PRIu64 "\n", workload_runs );
        fprintf( fp, "# calibrated ticks per second: %" PRIu64 "\n", ticks_per_second );
        fprintf( fp, "# Requested kernel duration: %d secs\n", 
                        benchmark_duration );
        fprintf( fp, "# Requested work iteration length: %d usecs.\n", usecs );
        fprintf( fp, "# Local Calibration duration: %.2lf secs\n", 
                        calibration_avg_local );
        fprintf( fp, "# Local work collection time: %.2lf\n",
                        collection_avg_local );
        fprintf( fp, "# index\t duration (nsec.)\n" );

        for( int i = 0 ; i < d->n ; i++ ){
                fprintf( fp, "%d\t %" PRIu64 "\n",
                                d->list[ i ].fwq.indx,
                                ( uint64_t )( ceil( 1.0e9 *
                                            ( ( double )( d->list[ i ].fwq.delta ) /
                                            ( double )( ticks_per_second ) ) ) ) );
        }

        return 0;
}

void
fwq_teardown( void *p )
{
        int rank, size;
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

int
fwq_setup( void *v, void *priv )
{

        dlist_t *d = NULL;
        fwq_params_t *f = NULL;
        int rank, size;
        long dpoints = 0;

        assert( v != NULL );

        d = ( dlist_t * )v;

        assert( priv != NULL );

        if( _workloads_initialized != 0 ){
                fprintf( stderr, "Calling setup() when already initialized\n" );
                return -1;
        }

        f = ( fwq_params_t * )priv;

        calibration_start = PMPI_Wtime();

        if( ticks_per_second == 0 ){
                ticks_per_second = get_ticks_per_second( 200 );
        }

        assert( ticks_per_second != 0 );

        benchmark_duration = d->duration;
        workload_verbose = d->verbose;

        usecs = f->usecs;

        assert( benchmark_duration != 0 );

        dpoints = ( long )( ( ( double )( benchmark_duration ) / 
                              ( double )( usecs * 1.0e-6 ) ) + 
                          ( ( ( double )( benchmark_duration ) / 
                              ( double )( usecs * 1.0e-6 ) ) * 0.2 ) );

        d->list = malloc( sizeof( datapoint_t ) * dpoints );
        assert( d->list != NULL );

        d->n = 0;
        d->max = dpoints;

        MPI_CHECK( PMPI_Comm_dup( f->comm, &workload_comm ) );
        MPI_CHECK( PMPI_Comm_rank( workload_comm, &rank ) );
        MPI_CHECK( PMPI_Comm_size( workload_comm, &size ) );

        if( ( workload_verbose ) && ( rank == 0 ) ){
                fprintf( stderr, "# Requested workload duration: %d usecs\n",
                                 usecs );
        }

        assert( benchmark_duration != 0 );

        /*
         * calibrate how often we need to call the workload macro in
         * order to get a minimal (noiseless) phase for the requested
         * number of usecs -kbf
         */
        if( ( workload_verbose ) && ( rank == 0 ) ){
                fprintf( stdout, "Calibrating FWQ workload \n" );
        }

        workload_runs = tune_workload( usecs );

        calibration_end = PMPI_Wtime();

        if( workload_verbose ){
                fprintf( stderr, "# [%d]: Workload runs: %" PRIu64 "\n",
                                 rank, workload_runs );
        }

        _workloads_stats_generated = 0;
        _workloads_initialized = 1;

        return 0;
}

int
fwq_run( void *v, void *private)
{
        dlist_t *d = NULL;
        double the_start, the_end;
        uint64_t start, end;
        int rank, size;
        long sample = 0;
        INIT_SCALAR_WORKLOAD;

        if( _workloads_initialized != 1 ){
                fprintf( stderr, "Calling run() when !initialized\n" );
                return -1;
        }

        assert( v != NULL );

        d = ( dlist_t * )v;

        MPI_CHECK( PMPI_Comm_rank( workload_comm, &rank ) );
        MPI_CHECK( PMPI_Comm_size( workload_comm, &size ) );

        assert( ticks_per_second != 0 );

        if( workload_verbose ){
                fprintf( stderr, "# [%d]: Starting FWQ workload tests\n",
                                rank );
        }

        d->run_start = the_start = collection_start = PMPI_Wtime();
        the_end = the_start + benchmark_duration;

        while( 1 ){
                start = rdtsc();

                for( uint64_t i = 0 ; i < workload_runs ; i++ ){
                        TEN( SCALAR_WORKLOAD );
                }

                end = rdtsc();

                d->list[ sample ].fwq.delta = end - start;
                d->list[ sample ].fwq.indx = sample;

                if( ++sample >= d->max )
                        abort();

                if( PMPI_Wtime() > the_end )
                        break;
        }

        d->run_end = collection_end = PMPI_Wtime();

        FINALIZE_SCALAR_WORKLOAD;

        if( workload_verbose ){
                fprintf( stderr, "# [%d]: FWQ workload test completed\n",
                                 rank );
        }

        d->n = sample;

        collection_avg_local += ( collection_end - collection_start );

        _workloads_stats_generated = 0;

        return sample;
}
