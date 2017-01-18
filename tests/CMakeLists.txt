
include( use_c99 )
use_c99()

# MPI Stuff
find_package( MPI REQUIRED )
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

include_directories( ${CMAKE_SOURCE_DIR}/src )

# A simple link test ----
add_executable( link_test link_test.c )
target_link_libraries( link_test workloads )

if( MPI_COMPILE_FLAGS )
        set_target_properties( link_test PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" )
endif()

if( MPI_LINK_FLAGS )
        set_target_properties( link_test PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}" )
endif()

#define a macro to simplify adding tests
macro(  do_test cmd arg result )
        add_test( workloads-${cmd} ${MPIEXEC} ${CMAKE_BINARY_DIR}/tests/${cmd} ${arg} )
        set_tests_properties( workloads-${cmd}
                PROPERTIES PASS_REGULAR_EXPRESSION ${result}
        )
endmacro( do_test )

do_test( link_test "" "*")
