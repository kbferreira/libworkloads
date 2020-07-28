#include "selfish.h"
#include "checks.h"
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <mpi.h>
#include <unistd.h>
#include <math.h>
#include "rdtsc.h"
#include "timing.h"
#include "workloads.h"

#define KERNEL_NAME "Selfish"
#define PROBLEM_DESC "rtsc() threshold timing"

static uint64_t threshold; /* in nanoseconds */
static uint64_t threshold_cycles; /* in cycles on the system */

static double collection_start, collection_end, collection_min,
              collection_max, collection_avg, collection_avg_local = 0.0;

static double calibration_start, calibration_end, calibration_avg,
              calibration_min, calibration_max, calibration_avg_local;
static long datapoints_local = 0, datapoints_min, datapoints_max;
static double datapoints_avg;

static int
do_stats( MPI_Comm this_comm )
{
        int rank, size;

        MPI_CHECK( PMPI_Comm_rank( this_comm, &rank) );
        MPI_CHECK( PMPI_Comm_size( this_comm, &size) );

        MPI_CHECK( PMPI_Allreduce( &datapoints_local, &datapoints_min, 1,
                                  MPI_LONG, MPI_MIN, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &datapoints_local, &datapoints_max, 1,
                                  MPI_LONG, MPI_MAX, this_comm ) );
        datapoints_avg = ( double )datapoints_local;
        MPI_CHECK( PMPI_Allreduce( MPI_IN_PLACE, &datapoints_avg, 1,
                                  MPI_DOUBLE, MPI_SUM, this_comm ) );
        datapoints_avg = datapoints_avg / ( double )size;

        collection_avg = collection_avg_local;
        MPI_CHECK( PMPI_Allreduce( &collection_avg_local, &collection_min, 1,
                                  MPI_DOUBLE, MPI_MIN, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &collection_avg_local, &collection_max, 1,
                                  MPI_DOUBLE, MPI_MAX, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &collection_avg_local, &collection_avg, 1,
                                  MPI_DOUBLE, MPI_SUM, this_comm ) );
        collection_avg = collection_avg / size;

        calibration_avg = calibration_avg_local = calibration_end - calibration_start;
        MPI_CHECK( PMPI_Allreduce( &calibration_avg_local, &calibration_min, 1,
                                  MPI_DOUBLE, MPI_MIN, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &calibration_avg_local, &calibration_max, 1,
                                  MPI_DOUBLE, MPI_MAX, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &calibration_avg_local, &calibration_avg, 1,
                                  MPI_DOUBLE, MPI_SUM, this_comm ) );
        calibration_avg = calibration_avg / size;

        _workloads_stats_generated = 1;

        return 0;

}

void
selfish_summary( FILE *fp )
{
        assert( fp != NULL );

        if( !_workloads_stats_generated )
                do_stats( workload_comm );

        fprintf( fp, "\nWorkload: %s.\nProblem description: %s\n", KERNEL_NAME,
                     PROBLEM_DESC );
        fprintf( fp, "Thresold for event: %" PRIu64 " nsecs (cycles: %" PRIu64 ")\n",
                      threshold, threshold_cycles );
        fprintf( fp, "Calibration Times:\t [ %.2lf, %.2lf, %.2lf ] secs\n",
                        calibration_min, calibration_avg, calibration_max );
        fprintf( fp, "Data Collection Times:\t [ %.2lf, %.2lf, %.2lf ] secs\n",
                        collection_min, collection_avg, collection_max );
        fprintf( fp, "Datapoints:\t\t [ %ld, %.2lf, %ld ]\n",
                        datapoints_min, datapoints_avg, datapoints_max );
        
        return;
}

int
selfish_write( FILE *fp, void *v, void *p )
{
        uint64_t start;
        dlist_t *d = NULL;

        assert( v != NULL );

        d = ( dlist_t *)v;

        assert( benchmark_duration != 0 );

        /*
         * We use ticks_per_second paramer, so make sure
         * it has been properly set
         */
        assert( ticks_per_second != 0 );

        if( !_workloads_stats_generated )
                do_stats( workload_comm );
        
        if( fp == NULL )
                fp = stdout;

        fprintf( fp, "# Selfish data run [ %.6lf %.6lf ]\n",
                                d->run_start - _workloads_epoch,
                                d->run_end - _workloads_epoch );
        fprintf( fp, "# calibrated ticks per second: %" PRIu64 "\n", ticks_per_second );
        fprintf( fp, "# Requested duration: %d secs\n", 
                        benchmark_duration );
        fprintf( fp, "# Detour threshold: %" PRIu64 " nsecs (cycles: %" PRIu64 ")\n",
                        threshold, threshold_cycles );
        fprintf( fp, "# Data Collection Times: local %.2lf [%.2lf, %.2lf, %.2lf] secs\n",
                        collection_avg_local, collection_min, collection_avg,
                        collection_max );
        fprintf( fp, "# tstamp\t duration (nsec)\n" );

        start = d->list[ 0 ].s.tstamp;

        for( int i = 0; i < d->n; i++ ){
                fprintf( fp, "%" PRIu64 "\t %" PRIu64 "\n",
                         ( uint64_t )( ceil( 1.0e9 * 
                                         ( ( double )( d->list[ i ].s.tstamp - start ) / 
                                           ( double )( ticks_per_second ) ) ) ), 
                         ( uint64_t )( ceil( 1.0e9 * 
                                         ( ( double )( d->list[ i ].s.delta ) / 
                                           ( double )( ticks_per_second ) ) ) ) );
        }

        d->n = 0;

        return 0;
}

void
selfish_teardown( void *p )
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
                d->n = d->max = 0;
        }

        MPI_CHECK( PMPI_Comm_free( &workload_comm ) );

        _workloads_initialized = 0;

        return;
}

int
selfish_setup( void *v, void *p )
{
        dlist_t *d = NULL;
        selfish_params_t *params = NULL;

        assert( v != NULL );
        assert( p !=  NULL );

        if( _workloads_initialized != 0 ){
                fprintf( stderr, "Calling setup() when already initialized\n" );
                return -1;
        }

        d = ( dlist_t *)v;
        /*
         * We use ticks_per_second, so make sure it is
         * set properly
         *
         * FIXME: This calibration time is not tracked for output -kbf
         */
        if( ticks_per_second == 0 ){
                calibration_start = PMPI_Wtime();
                ticks_per_second = get_ticks_per_second( 200 );
                calibration_end = PMPI_Wtime();
        }

        assert( ticks_per_second != 0 );

        benchmark_duration = d->duration;
        workload_verbose = d->verbose;

        assert( benchmark_duration != 0 );

        params = ( selfish_params_t * )p;

        d->max = params->size;

        assert( d->max > 0 );

        d->list = malloc( sizeof( datapoint_t ) * d->max );
        assert( d->list != NULL );

        d->n = 0;

        threshold = params->threshold;

        threshold_cycles = ( uint64_t )( ( double )( threshold ) / 1.0e9 * 
                           ( double )( ticks_per_second ) );

        MPI_CHECK( PMPI_Comm_dup( params->comm, &workload_comm ) );

        _workloads_stats_generated = 0;
        _workloads_initialized = 1;

        return 0;
}

int
selfish_run( void *v, void *t )
{
        double the_start, the_end;
        uint64_t start, now;
        int rank, size;
        long sample = 0;
        dlist_t *d = NULL;

        if( _workloads_initialized != 1 ){
                fprintf( stderr, "Calling run() when !initialized\n" );
                return -1;
        }

        assert( v != NULL  );

        d = ( dlist_t *)v;

        assert( ticks_per_second != 0 );
        assert( benchmark_duration != 0 );

        /*
         * FIXME: This timing stuff should be a bit
         * better abstracted and not directly use MPI
         * as that might notbe suffucent resilution
         */
        d->run_start = the_start = collection_start = PMPI_Wtime();
        the_end = the_start + benchmark_duration;

        sample = d->n;

        start = rdtsc();
        while( 1 ){
                now = rdtsc();
                uint64_t delta = now - start;
                start = now;

                if( PMPI_Wtime() > the_end )
                        break;

                if( delta < threshold_cycles )
                        continue;

                /* 
                 * An interruption
                 */
                d->list[ sample ].s.tstamp = now;
                d->list[ sample ].s.delta = delta;

                if( ++sample >= d->max ){
                        d->list = realloc( d->list, sizeof( datapoint_t ) * d->max * 2 );
                        if( d->list == NULL ) {
                          // !!!! DEBUG !!!!
                          printf("Failed to ALLOCATE %d (%d x %d x %d) bytes\n", 
                                 sizeof( datapoint_t ) * d->max * 2, sizeof( datapoint_t ), d->max, 2);
                          fflush(stdout);
                        }
                        assert( d->list != NULL );
                        d->max = d->max * 2;
                }

        }

        d->run_end = collection_end = PMPI_Wtime();

        collection_avg_local += ( collection_end - collection_start );

        datapoints_local += d->n = sample;

        _workloads_stats_generated = 0;

        return sample;
}
