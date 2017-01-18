#include "simple-noise.h"
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include "rdtsc.h"
#include "checks.h"
#include "workloads.h"
#include "timing.h"
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <sys/time.h>

static int freq_HZ = 0;
static int duration_usecs;

static uint64_t workload_runs;

int simple_noise_schedule = 0;

static int opt_output;

#define KERNEL_NAME "Simple OS Noise emulation"
#define PROBLEM_DESC "Requires frequent calls into MPI library.  Workload is a cache-friendly workload from netgauge"

static double calibration_start, calibration_end, calibration_avg,
              calibration_avg_local, calibration_min, calibration_max;
static double event_count = 0.0, duration_accum = 0.0, running_HZ_avg_local,
              running_HZ_avg, running_HZ_min, running_HZ_max, running_duration_avg,
              running_duration_avg_local, running_duration_min, running_duration_max;

static double simple_noise_start;

static long int irq_count = 0;

static void
noise_sigalrm_handler( int signo, siginfo_t *info, void *context )
{
        simple_noise_schedule++;

        irq_count++;

        return;
}

static int
init_noise_alrm( void )
{
        struct sigaction my_action;
        struct itimerval timer;
        double interarrival;

        assert( freq_HZ != 0 );

        interarrival = ( 1.0 / ( double )freq_HZ );
        memset( &my_action, 0, sizeof( my_action ) );

        my_action.sa_flags = SA_SIGINFO;
        my_action.sa_sigaction = &noise_sigalrm_handler;
        SYSCALL_CHECK( sigaction( SIGALRM, &my_action, NULL ) );

        timer.it_interval.tv_sec = timer.it_value.tv_sec = ( int )( interarrival );
        timer.it_interval.tv_usec = timer.it_value.tv_usec = 
                ( int )( ( interarrival - ( double )( ( int )( interarrival ) ) ) * 
                                1.0e6 );

        SYSCALL_CHECK( setitimer( ITIMER_REAL, &timer, NULL ) );

        return 0;
}

static int
cancel_noise_alrm( void )
{
        int rc;
        struct sigaction my_action;

        memset( &my_action, 0, sizeof( my_action ) );
        my_action.sa_handler = SIG_IGN;

        SYSCALL_CHECK( rc = sigaction( SIGALRM, &my_action, NULL ) );

        return rc;
}
static int
do_stats( MPI_Comm this_comm )
{
        int rank, size;

        MPI_CHECK( PMPI_Comm_rank( this_comm, &rank ) );
        MPI_CHECK( PMPI_Comm_size( this_comm, &size ) );

        assert( size != 0 );

        calibration_avg = calibration_avg_local = calibration_end - calibration_start;
        MPI_CHECK( PMPI_Allreduce( &calibration_avg_local, &calibration_min, 1,
                                  MPI_DOUBLE, MPI_MIN, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &calibration_avg_local, &calibration_max, 1,
                                  MPI_DOUBLE, MPI_MAX, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &calibration_avg_local, &calibration_avg, 1,
                                  MPI_DOUBLE, MPI_SUM, this_comm ) );
        calibration_avg = calibration_avg / size;

        running_HZ_avg = running_HZ_avg_local = 
                event_count / ( PMPI_Wtime() - simple_noise_start );
        MPI_CHECK( PMPI_Allreduce( &running_HZ_avg_local, &running_HZ_min, 1,
                                  MPI_DOUBLE, MPI_MIN, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &running_HZ_avg_local, &running_HZ_max, 1,
                                  MPI_DOUBLE, MPI_MAX, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &running_HZ_avg_local, &running_HZ_avg, 1,
                                  MPI_DOUBLE, MPI_SUM, this_comm ) );
        running_HZ_avg = running_HZ_avg / ( double )size;

        running_duration_avg = running_duration_avg_local = 
                ( event_count != 0.0 )? duration_accum / event_count :
                                        0.0;
        MPI_CHECK( PMPI_Allreduce( &running_duration_avg_local, &running_duration_min,
                                1, MPI_DOUBLE, MPI_MIN, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &running_duration_avg_local, &running_duration_max,
                                1, MPI_DOUBLE, MPI_MAX, this_comm ) );
        MPI_CHECK( PMPI_Allreduce( &running_duration_avg_local, &running_duration_avg,
                                1, MPI_DOUBLE, MPI_SUM, this_comm ) );
        running_duration_avg = running_duration_avg / ( double )size;

        _workloads_stats_generated = 1;

        return 0;

}

void
simple_noise_summary( FILE *fp )
{
        assert( fp != NULL );

        if( !_workloads_stats_generated )
                do_stats( workload_comm );

        fprintf( fp, "\nWorkload: %s.\nProblem description: %s\n", KERNEL_NAME,
                     PROBLEM_DESC );
        fprintf( fp, "Requested Frequency:\t\t\t %10d Hz\n", freq_HZ );
        fprintf( fp, "Requested Duration:\t\t\t %10d usecs\n", duration_usecs );
        fprintf( fp, "Avg. Frequency:\t [ %.2lf, %.2lf, %.2lf ] HZ\n", 
                        running_HZ_min, running_HZ_avg, running_HZ_max );
        fprintf( fp, "Avg. Duration:\t [ %.2lf, %.2lf, %.2lf ] usecs\n",
                        running_duration_min, running_duration_avg,
                        running_duration_max );

        return;
}
int
simple_noise_write( FILE  *fp, void *v, void *p )
{
        dlist_t *d = NULL;

        assert( v != NULL );

        d = ( dlist_t * )v;

        /*
         * We use ticks_per_second parameter, so make
         * sure it as been properly set -kbf
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
                        PMPI_Wtime() - _workloads_epoch );
        fprintf( fp, "# calibrated ticks per second: %" PRIu64 "\n", ticks_per_second );
        fprintf( fp, "# Requested Frequency:\t\t\t %10d Hz\n", freq_HZ );
        fprintf( fp, "# Requested Duration:\t\t\t %10d usecs\n", duration_usecs );
        fprintf( fp, "# Avg. Frequency:\t [ %.2lf, %.2lf, %.2lf ] HZ\n", 
                        running_HZ_min, running_HZ_avg, running_HZ_max );
        fprintf( fp, "# Local Avg. Frequency:\t %.2lf Hz\n", running_HZ_avg_local );
        fprintf( fp, "# Avg. Duration:\t [ %.2lf, %.2lf, %.2lf ] usecs\n",
                        running_duration_min, running_duration_avg,
                        running_duration_max );
        fprintf( fp, "# Local Avg. Duration:\t %.2lf usecs\n",
                        running_duration_avg_local );
        fprintf( fp, "# Local calibration time: %.2lf\n", calibration_avg_local );
        fprintf( fp, "# Optimition avoidance (ignore): %d\n", opt_output );
        fprintf( fp, "# IRQ count: %ld\n", irq_count );
        fprintf( fp, "# Event count: %.2lf, Duration Accum: %.2lf\n",
                        event_count, duration_accum );

        return 0;
}

void
simple_noise_teardown( void *p )
{
        int rank, size;
        dlist_t *d = NULL;

        assert( p != NULL );

        if( _workloads_initialized != 1 ){
                fprintf( stderr, "Calling teardown() when !initialized\n" );
                return;
        }

        cancel_noise_alrm();

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
simple_noise_setup( void *v, void *priv )
{
        dlist_t *d = NULL;
        simple_noise_params_t *f = NULL;
        int rank, size;
        long dpoints = 0;

        assert( v != NULL );

        d = ( dlist_t * )v;

        assert( priv != NULL );

        if( _workloads_initialized != 0 ){
                fprintf( stderr, "Calling setup() when already initialized\n" );
                return -1;
        }

        f = ( simple_noise_params_t * )priv;

        calibration_start = PMPI_Wtime();

        if( ticks_per_second == 0 ){
                ticks_per_second = get_ticks_per_second( 200 );
        }

        assert( ticks_per_second != 0 );

        workload_verbose = d->verbose;

        freq_HZ = f->freq_HZ;
        duration_usecs = f->duration_usecs;

        d->list = NULL; /* XXX: NO DATA COLLECTED */
        d->n = 0;
        d->max = 0;

        MPI_CHECK( PMPI_Comm_dup( f->comm, &workload_comm ) );
        MPI_CHECK( PMPI_Comm_rank( workload_comm, &rank ) );
        MPI_CHECK( PMPI_Comm_size( workload_comm, &size ) );

        if( ( workload_verbose ) && ( rank == 0 ) ){
                fprintf( stderr, "# Requested frequency: %d Hz, duration: %d usecs\n",
                                 freq_HZ, duration_usecs );
        }

        /*
         * calibrate how often we need to call the workload macro in
         * order to get a minimal (noiseless) phase for the requested
         * number of usecs -kbf
         */
        if( ( workload_verbose ) && ( rank == 0 ) ){
                fprintf( stdout, "Calibrating Simple Noise workload \n" );
        }

        /*
         * FIXME: Should I really be calling MPI_Wtime()
         *        directly? -kbf
         */
        workload_runs = tune_workload_var( duration_usecs, &opt_output );
        calibration_end = PMPI_Wtime();

        simple_noise_schedule = 0;

        d->run_start = simple_noise_start = PMPI_Wtime();

        init_noise_alrm();

        _workloads_stats_generated = 0;
        _workloads_initialized = 1;

        return 0;
}

int
simple_noise_run( void *v, void *private)
{
        dlist_t *d = NULL;
        int rank, size;
        uint64_t start, end, elapsed;
        uint64_t ticks_per_cycle;
        uint64_t wruns;
        uint64_t divider = 1000;
        double nusecs;
        INIT_SCALAR_WORKLOAD_VAR( opt_output );

        do{
                wruns = workload_runs / divider;
                divider = divider / 10;
        } while( wruns == 0 );

        assert( wruns > 0 );

        start = rdtsc();

        if( _workloads_initialized != 1 ){
                fprintf( stderr, "Calling run() when !initialized\n" );
                return -1;
        }

        assert( v != NULL );

        d = ( dlist_t * )v;

        assert( ticks_per_second != 0 );

        MPI_CHECK( PMPI_Comm_rank( workload_comm, &rank ) );
        MPI_CHECK( PMPI_Comm_size( workload_comm, &size ) );

        if( workload_verbose ){
                fprintf( stderr, "# [ %d ] : Calibrated workload runs: %" PRIu64 "\n",
                                 rank, workload_runs );
        }

        ticks_per_cycle = ( uint64_t )( ceil( ( ( double )ticks_per_second /
                                        1.0e6 ) * ( double )duration_usecs ) );

        if( workload_verbose ){
                fprintf( stderr, "# [%d]: Generating Noise event\n",
                                rank );
        }

        event_count += 1.0;

        do{
                for( uint64_t i = 0 ; i < wruns ; i++ ){
                        TEN( SCALAR_WORKLOAD );
                }
                end = rdtsc();
                elapsed = end - start;
        } while( elapsed < ticks_per_cycle );

        nusecs = ( ( double )elapsed / ( double )ticks_per_second ) * 1.0e6;
        duration_accum += nusecs;

        if( workload_verbose ){
                fprintf( stderr, "# [ %d ] : duration %.2lf usecs\n",
                                 rank, nusecs );
        }

        FINALIZE_SCALAR_WORKLOAD_VAR( opt_output );

        if( workload_verbose ){
                fprintf( stderr, "# [%d]: Simple-noise event completed\n",
                                 rank );
        }

        _workloads_stats_generated = 0;

        return 0;
}
