#ifndef _RDTSC_H
#define _RDTSC_H (1)

#include <stdlib.h>
#include <stdint.h>

typedef uint64_t ticks_t;

#if defined(__i386__)

static __inline__ ticks_t rdtsc(void)
{
  unsigned long long int x;
     __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
     return x;
}
#elif defined(__x86_64__)


static __inline__ ticks_t rdtsc(void)
{
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (ticks_t)lo)|( ((ticks_t)hi)<<32 );
}

#elif defined(__aarch64__)

static __inline__ ticks_t rdtsc(void)
{
  unsigned long long t1;
  __asm__ __volatile__( \
                       "isb                      \n"          \
                       "mrs %0, cntvct_el0       \n"          \
                       : "=r"(t1) : : "memory"); 
  return ( (ticks_t)t1 );
}

#elif defined(__powerpc__)
static inline ticks_t rdtsc(void)
{
  unsigned long long int result=0;
  unsigned long int upper, lower,tmp;
  __asm__ __volatile__(
                "0:                \n\t"
                "\tmftbu   %0      \n\t"
                "\tmftb    %1      \n\t"
                "\tmftbu   %2      \n\t"
                "\tcmpw    %2,%0   \n\t"
                "\tbne     0b      \n\t"
                : "=r"(upper),"=r"(lower),"=r"(tmp)
                );
  result = upper;
  result = result<<32;
  result = result|lower;
  return(result);
}
#else
#error "No tick counter is available!"
#endif /* ARCH */

#endif /* _RDTSC_H */
