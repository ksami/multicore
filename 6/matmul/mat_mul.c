#include <stdio.h>
#include <stdlib.h>
#include "timers.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

#define NDIM 4096

float a[NDIM][NDIM];
float b[NDIM][NDIM];
float c[NDIM][NDIM];


int main(int argc, char* argv[]) {
    int i, j, k=1;
    int result=0;
    int cnt_threads = strtol(argv[1], NULL, 10);

    timer_init();

    for( i = 0; i < NDIM; i++ )
    {
        for( j = 0; j < NDIM; j++ )
        {
            a[i][j] = k;
            b[i][j] = k;
            k++;
        }
    }

    timer_start(1);

    #pragma omp parallel num_threads(cnt_threads) shared(a, b, c) private(i, j, k)
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

    timer_stop(1);
    printf("Time elapsed : %lf sec\n", timer_read(1));

    return 0;
}
