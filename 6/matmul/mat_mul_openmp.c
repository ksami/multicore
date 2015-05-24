#include <stdio.h>
#include <stdlib.h>
#include "timers.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

#define NDIM 4

// float a[NDIM][NDIM];
// float b[NDIM][NDIM];
// float c[NDIM][NDIM];

// void hello(void);

int main(int argc, char* argv[]) {
    int i, j, k=1;
    int cnt_threads = strtol(argv[1], NULL, 10);

    // for( i = 0; i < NDIM; i++ )
    // {
    //     for( j = 0; j < NDIM; j++ )
    //     {
    //         a[i][j] = k;
    //         b[i][j] = k;
    //         k++;
    //     }
    // }



    #pragma omp parallel num_threads(cnt_threads)
    {
        #pragma omp for
        // for( i = 0; i < NDIM; i++ )
        // {
        //     for( j = 0; j < NDIM; j++ )
        //     {
                for( k = 0; k < NDIM; k++ )
                {
                    // c[i][j] += a[i][k] * b[k][j];
                    printf("id: %d\n", omp_get_thread_num());
                }
        //     }
        // }
    }

    return 0;
}

// void hello(void) {
//     int my_id, num_threads;
    
//     my_id = omp_get_thread_num();
//     num_threads = omp_get_num_threads();
    
//     printf("id: %d, num_threads: %d\n", my_id, num_threads);
// }
