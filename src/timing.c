#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#include "rdtsc.h"
#include <mpi.h>
#include <math.h>

uint64_t
get_ticks_per_second( int rounds )
{
        uint64_t t1, t2, *res = NULL;
        static uint64_t min = 0;
        double start, end;

        assert( rounds > 0 );

        if( min >  0 )
                return min;

        res = malloc( sizeof( uint64_t ) * rounds );
        assert( res != NULL );

        for( int i = 0 ; i < rounds ; i++ ){
                start = PMPI_Wtime();
                t1 = rdtsc();
                sleep( 1 );
                t2 = rdtsc();
                end = PMPI_Wtime();

#if 0
                // !!!! DEBUG !!!!
                printf("(get_ticks_per_second) TIMESTAMPS: t1 = %ld / t2 = %ld / t2-t1 = %ld / start = %f / end = %f\n", 
                       t1, t2, t2-t1, start, end); 
#endif

                res[ i ] = ( uint64_t )( ceil( ( ( double )t2 - ( double )t1 ) /
                                ( end - start ) ) );
        }

        min = res[ 0 ];

        for( int i = 0 ; i < rounds ; i++ ){
                if( min > res[ i ] )
                        min = res[ i ];
        }

        free( res );

        return min;
}
