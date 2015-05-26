#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "timers.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

#define NDIM 4096

float a[NDIM][NDIM];
float b[NDIM][NDIM];
float c[NDIM][NDIM];

static inline double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1.0e-6*tv.tv_usec;
}


int main(int argc, char* argv[]) {
    int i, j, k=1;
    double start, end;
    int cnt_threads = strtol(argv[1], NULL, 10);


    for( i = 0; i < NDIM; i++ )
    {
        for( j = 0; j < NDIM; j++ )
        {
            a[i][j] = k;
            b[i][j] = k;
            k++;
        }
    }

    start = get_time();

    #pragma omp parallel num_threads(cnt_threads) shared(a, b, c) private(i, j, k) collapse(3)
    {
        #pragma omp for
        for( i = 0; i < NDIM; i++ )
        {
            for( j = 0; j < NDIM; j++ )
            {
                for( k = 0; k < NDIM; k++ )
                {
                    #pragma omp atomic
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    end = get_time();

    printf("Time elapsed : %lf sec\n", end-start);

    return 0;
}
