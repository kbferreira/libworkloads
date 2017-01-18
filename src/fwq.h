#ifndef FWQ_H
#define FWQ_H ( 1 )

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <mpi.h>

typedef struct {
        int usecs;
        MPI_Comm comm;
} fwq_params_t;

typedef struct {
        uint64_t delta;
        int indx;
} fwq_t;

int fwq_setup( void *, void * );
int fwq_run( void *, void * );
int fwq_write( FILE *, void *, void * );
void fwq_summary( FILE * );
void fwq_teardown( void * );
#endif /* FWQ_H */
