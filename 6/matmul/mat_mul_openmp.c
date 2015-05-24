#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

void hello(void);

void main(int argc, char* argv[]) {
    int cnt_threads = strtol(arg[1], NULL, 10);

    #pragma omp parallel num_threads(cnt_threads)
    hello();

    return 0;
}

void hello(void) {
    int my_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    
    printf(“Hello world! %d %d\n”, my_id, num_threads);
}