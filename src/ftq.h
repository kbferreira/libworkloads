#ifndef FTQ_H
#define FTQ_H ( 1 )

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <mpi.h>

typedef struct {
        int usecs;
        MPI_Comm comm;
} ftq_params_t;

typedef struct {
        uint64_t iters;
        int indx;
} ftq_t;

int ftq_setup( void *, void * );
int ftq_run( void *, void * );
int ftq_write( FILE *, void *, void * );
void ftq_summary( FILE * );
void ftq_teardown( void * );
#endif /* FTQ_H */
