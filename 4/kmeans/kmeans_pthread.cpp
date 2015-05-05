
#include "kmeans.h"

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <pthread.h>

#define NUM_THREADS 4

int iteration_n;
int class_n;
int data_n;
Point* centroids;
Point* data;
int* partitioned;
int* count;


void *kmeans_t_assign(void *thread_id)
{
    int* id;

    printf("debug: initializing thread\n");

    id = (int *) thread_id;

    printf("thread %d\n", id);

    pthread_exit(NULL);
}


void kmeans(int aiteration_n, int aclass_n, int adata_n, Point* acentroids, Point* adata, int* apartitioned)
{
    iteration_n = aiteration_n;
    class_n = aclass_n;
    data_n = adata_n;
    centroids = acentroids;
    data = adata;
    partitioned = apartitioned;
    // Count number of data in each class
    count = (int*)malloc(sizeof(int) * aclass_n);
    
    pthread_t threads[NUM_THREADS];
    int retcode;
    int i, t;

    // Iterate through number of interations
    for (i = 0; i < iteration_n; i++)
    {
        printf("Iteration %d\n", i);

        // Assignment //
        for(t=0; t<NUM_THREADS; t++)
        {
            retcode = pthread_create(&threads[t], NULL, kmeans_t_assign, (void*) t);

            printf("debug: %d\n", t);

            if(retcode)
            {
                printf("ERROR: return code from pthread_create() for id=%d is %d\n", t, retcode);
                exit(-1);
            }
        }
        for(t=0; t<NUM_THREADS; t++)
        {
            printf("debug: joining %d\n",t);
            pthread_join(threads[t], NULL);
        }
    }
}

