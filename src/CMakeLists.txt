
option( DEBUG_PRINT "DEBUG" OFF )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} )

include( use_c99 )
use_c99()

# MPI Stuff
find_package( MPI REQUIRED )
include_directories( SYSTEM ${MPI_INCLUDE_PATH} )

add_library( workloads STATIC
             selfish.c timing.c stencil.c workloads.c fwq.c ftq.c
             simple-noise.c )
target_link_libraries( workloads ${MPI_C_LIBRARIES} )

if( MPI_C_COMPILE_FLAGS )
        set_target_properties( workloads PROPERTIES COMPILE_FLAGS
                "${MPI_C_COMPILE_FLAGS}" )
endif()

if( MPI_C_LINK_FLAGS )
        set_target_properties( workloads PROPERTIES LINK_FLAGS
                "${MPI_C_LINK_FLAGS}" )
endif()

set( CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}"
     "${MPI_C_LIBRARIES}" )
set( CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}"
     "${MPI_C_COMPILE_FLAGS}" )
set( CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}"
     "${MPI_C_INCLUDE_PATH}" )

include( CheckCSourceCompiles )

unset( HAVE_NB_ALLREDUCE CACHE )

check_c_source_compiles(
"
#include <mpi.h>
#include <stdlib.h>

int main( void )
{
        MPI_Iallreduce( NULL, NULL, 0, MPI_INT, MPI_SUM,
                        MPI_COMM_WORLD, MPI_REQUEST_NULL );
        return 0;
}
" HAVE_NB_ALLREDUCE )

if( HAVE_NB_ALLREDUCE )
        message( STATUS "Have MPI_Iallreduce(), we are good" )
else()
        message( FATAL_ERROR "Needed MPI_Iallreduce() (MPI 3.X+) Not Found" )
endif()

install( TARGETS workloads DESTINATION lib )
install( FILES ftq.h fwq.h selfish.h stencil.h workloads.h simple-noise.h DESTINATION include )
