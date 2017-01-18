#ifndef SELFISH_H
#define SELFISH_H ( 1 )

#include <stdint.h>
#include <stdio.h>
#include <mpi.h>

typedef struct SELFISH_T {
        uint64_t tstamp;
        uint64_t delta;
} selfish_t;

typedef struct {
        long      size;
        uint64_t threshold;
        MPI_Comm comm;
} selfish_params_t;

int selfish_setup( void *, void * ); 
int selfish_run( void *, void * ); 
int selfish_write( FILE *, void *, void * ); 
void selfish_summary( FILE * ); 
void selfish_teardown( void * ); 
#endif /* SELFISH_H */
