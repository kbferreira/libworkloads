#ifndef STENCIL_H
#define STENCIL_H ( 1 )

#include <stdint.h>
#include <stdio.h>
#include <mpi.h>

typedef struct {
        int ndims;
        int npx, npy, npz;
        int usecs;
        MPI_Comm comm;
} stencil_params_t;

typedef struct {
        uint64_t delta;
        uint64_t wait;
        int indx;
} stencil_t;

int stencil_setup( void *, void * );
int stencil_run( void *d, void * );
int stencil_write( FILE *, void *, void * );
void stencil_summary( FILE * );
void stencil_teardown( void * );

#define INIT_STENCIL                                            \
        /*                                                      \
         * Create a temp file for write to avoid compiler       \
         * optimizations -kbf                                   \
         */                                                     \
        memset( tmpfname, 0, sizeof( tmpfname ) );              \
        snprintf( tmpfname, FNAME_MAX,                          \
                        "/tmp/jitter-stencil-%d-XXXXXX",        \
                        rank );                                 \
        opt_fd = mkstemp( tmpfname );                           \
        if( opt_fd < 1 ){                                       \
                perror( "Cannot create temp output file" );     \
                abort();                                        \
        }                                                       \
        unlink( tmpfname );

#define FINALIZE_STENCIL( value ) \
        dprintf( opt_fd, "# Random output (no compiler opt): %i\n", value); \
        close( opt_fd );

#endif /* STENCIL_H */
