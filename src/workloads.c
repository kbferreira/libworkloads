#include "fwq.h"
#include "ftq.h"
#include "selfish.h"
#include "stencil.h"
#include "simple-noise.h"
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "rdtsc.h"
#include "checks.h"
#include "workloads.h"
#include "timing.h"
#include <mpi.h>

uint64_t ticks_per_second = 0;
int benchmark_duration = 0;
int workload_verbose = 0;
MPI_Comm workload_comm;

const char* _workloads_bnames[] = { "selfish", "stencil", "fwq", "ftq", "simple-noise" };

int _workloads_initialized = 0;
int _workloads_stats_generated = 0;

double _workloads_epoch = 0;

int
workloads_configure( dlist_t *d, type_t t )
{
        assert( d != NULL );

        assert( ( t > UNDEF ) && ( t < INVALID ) );

        _workloads_epoch = PMPI_Wtime();

        switch( t ){
                case SELFISH:
                        d->setup = selfish_setup;
                        d->run = selfish_run;
                        d->write = selfish_write;
                        d->summary = selfish_summary;
                        d->teardown = selfish_teardown;
                        d->d_type = SELFISH;
                        break;
                case STENCIL:
                        d->setup = stencil_setup;
                        d->run = stencil_run;
                        d->write = stencil_write;
                        d->summary = stencil_summary;
                        d->teardown = stencil_teardown;
                        d->d_type = STENCIL;
                        break;
                case FWQ:
                        d->setup = fwq_setup;
                        d->run = fwq_run;
                        d->write = fwq_write;
                        d->summary = fwq_summary;
                        d->teardown = fwq_teardown;
                        d->d_type = FWQ;
                        break;
                case FTQ:
                        d->setup = ftq_setup;
                        d->run = ftq_run;
                        d->write = ftq_write;
                        d->summary = ftq_summary;
                        d->teardown = ftq_teardown;
                        d->d_type = FTQ;
                        break;
                case SIMPLE_NOISE:
                        d->setup = simple_noise_setup;
                        d->run = simple_noise_run;
                        d->write = simple_noise_write;
                        d->summary = simple_noise_summary;
                        d->teardown = simple_noise_teardown;
                        d->d_type = SIMPLE_NOISE;
                        break;
                case UNDEF:
                case INVALID:
                        abort();
        }

        return 0;
}

uint64_t 
tune_workload( int usecs )
{
        double desired_ticks;
        uint64_t workcount, min, lmin;

        if( ticks_per_second == 0 ){
                get_ticks_per_second( 200 );
        }

        assert( ticks_per_second != 0 );

        desired_ticks = ( ticks_per_second / 1e6 ) * usecs;

        workcount = 1000;
        min = 0;
        INIT_SCALAR_WORKLOAD;

        while( ( min == 0 ) || 
               ( ( fabs( desired_ticks - min ) / desired_ticks ) > 0.005 ) )
        {
                uint64_t t1, t2;
                min = 0;

                for( int i = 0 ; i < 200 ; i++ ){
                        t1 = rdtsc();
                        for( int j = 0 ; j < workcount ; j++ ){
                                TEN( SCALAR_WORKLOAD );
                        }
                        t2 = rdtsc();

                        lmin = t2 - t1;

                        if( ( min == 0 ) || ( min > lmin ) ){
                                min = lmin;
                        }
                }

                workcount = ( uint64_t )( ceil( ( desired_ticks / ( double )min ) *
                                                ( double )workcount ) );
        }

        FINALIZE_SCALAR_WORKLOAD;

        return workcount;
}
uint64_t 
tune_workload_var( int usecs, int *output )
{
        double desired_ticks;
        uint64_t workcount, min, lmin;

        if( ticks_per_second == 0 ){
                get_ticks_per_second( 200 );
        }

        assert( ticks_per_second != 0 );

        desired_ticks = ( ticks_per_second / 1e6 ) * usecs;

        workcount = 1000;
        min = 0;
        INIT_SCALAR_WORKLOAD_VAR( (*output) );

        while( ( min == 0 ) || 
               ( ( fabs( desired_ticks - min ) / desired_ticks ) > 0.005 ) )
        {
                uint64_t t1, t2;
                min = 0;

                for( int i = 0 ; i < 200 ; i++ ){
                        t1 = rdtsc();
                        for( int j = 0 ; j < workcount ; j++ ){
                                TEN( SCALAR_WORKLOAD );
                        }
                        t2 = rdtsc();

                        lmin = t2 - t1;

                        if( ( min == 0 ) || ( min > lmin ) ){
                                min = lmin;
                        }
                }

                workcount = ( uint64_t )( ceil( ( desired_ticks / ( double )min ) *
                                                ( double )workcount ) );
        }

        FINALIZE_SCALAR_WORKLOAD_VAR( (*output) );

        return workcount;
}
