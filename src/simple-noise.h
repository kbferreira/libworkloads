#ifndef SIMPLE_NOISE_H
#define SIMPLE_NOISE_H ( 1 )

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <mpi.h>

typedef struct {
        int freq_HZ;
        int duration_usecs;
        MPI_Comm comm;
} simple_noise_params_t;

int simple_noise_setup( void *, void * );
int simple_noise_run( void *, void * );
int simple_noise_write( FILE *, void *, void * );
void simple_noise_summary( FILE * );
void simple_noise_teardown( void * );

extern int simple_noise_schedule;

#endif /* SIMPLE_NOISE_H */
