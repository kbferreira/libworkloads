#ifndef WORKLOAD_H
#define WORKLOAD_H ( 1 )

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include "selfish.h"
#include "stencil.h"
#include "ftq.h"
#include "fwq.h"
#include "simple-noise.h"
#include <mpi.h>

typedef enum DATA_T { UNDEF=-1, SELFISH=0, STENCIL, FWQ, FTQ, SIMPLE_NOISE,
                      INVALID } type_t;

typedef struct {
        union {
                selfish_t s;
                stencil_t c;
                fwq_t fwq;
                ftq_t ftq;
        };
} datapoint_t;

typedef struct DLIST_T {
        int ( *setup )( void *d, void *p );
        int ( *run )( void *d, void *p );
        int ( *write )( FILE *fp, void *d, void *p);
        void ( *summary )( FILE *fp );
        void ( *teardown )( void * );

        type_t d_type;
        long n;
        long max;
        datapoint_t *list;
        int verbose;
        int duration;

        double run_start;
        double run_end;
} dlist_t;

#define FNAME_MAX       ( 100 )

/*
 * Silly workload stuff doe fwq/ftq.  From netgauge -kbf
 */
#define INIT_SCALAR_WORKLOAD                                                    \
                        int _workload_a = rand(); int _workload_b = rand();     \
                        int _workload_n = rand(); int opt_fd = -1;              \
                        char tmpfname[ FNAME_MAX ];                             \
        /*                                                                      \
         * Create a temp file for write to avoid compiler                       \
         * optimizations -kbf                                                   \
         */                                                                     \
        memset( tmpfname, 0, sizeof( tmpfname ) );                              \
        snprintf( tmpfname, FNAME_MAX, "/tmp/jitter-workload-XXXXXX" );         \
        opt_fd = mkstemp( tmpfname );                                           \
        if( opt_fd < 1 ){                                                       \
                perror( "Cannot create temp output file" );                     \
                abort();                                                        \
        }                                                                       \
        unlink( tmpfname );

#define INIT_SCALAR_WORKLOAD_VAR( this_var )                                    \
                        int _workload_a = rand(); int _workload_b = rand();     \
                        int _workload_n = rand();                               \

#define SCALAR_WORKLOAD                                                         \
        _workload_a ^= _workload_a + _workload_a;                               \
        _workload_a ^= _workload_a + _workload_a + _workload_a;                 \
        _workload_a >>= _workload_b;                                            \
        _workload_a >>= _workload_a + _workload_a;                              \
        _workload_a ^= _workload_a + _workload_b;                               \
        _workload_a += ( _workload_a + _workload_a ) & 07;                      \
        _workload_a ^= _workload_n;                                             \
        _workload_b ^= _workload_a;                                             \
        _workload_a |= _workload_b;

#define TEN( A ) A A A A A A A A A A 

#define HUNDRED( A ) TEN( A ) TEN( A ) TEN( A ) TEN( A ) TEN( A ) \
                     TEN( A ) TEN( A ) TEN( A ) TEN( A ) TEN( A )

#define THOUSAND( A ) HUNDRED( A ) HUNDRED( A ) HUNDRED( A ) HUNDRED( A ) \
                      HUNDRED( A ) HUNDRED( A ) HUNDRED( A ) HUNDRED( A ) \
                      HUNDRED( A ) HUNDRED( A )

#define FINALIZE_SCALAR_WORKLOAD                                                \
        dprintf( opt_fd, "# Random output (no compiler opt): %i\n",             \
                         _workload_a + _workload_b );                           \
        close(  opt_fd );

#define FINALIZE_SCALAR_WORKLOAD_VAR( this_var )                                \
        this_var = _workload_a + _workload_b + _workload_n

uint64_t tune_workload( int usecs );
uint64_t tune_workload_var( int usecs, int *output );

extern uint64_t ticks_per_second;
extern int benchmark_duration; /* in seconds */
extern int workload_verbose;
extern MPI_Comm workload_comm;

extern int _workloads_initialized;
extern int _workloads_stats_generated;

extern double _workloads_epoch;

extern const char* _workloads_bnames[];

int workloads_configure( dlist_t *d, type_t t );

#endif /* WORKLOAD_H */
