
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
	int id;
	int data_i, class_i;
	// Temporal point value to calculate distance
	Point t;

	id = *((int *) thread_id);

	// Assignment step
	for (data_i = data_n*(id/NUM_THREADS); data_i < data_n*((id+1)/NUM_THREADS); data_i++) {
		printf("debug: thread %d, data_i %d/%d\n",id,data_i,data_n);

		float min_dist = DBL_MAX;
	
		for (class_i = 0; class_i < class_n; class_i++) {
			t.x = data[data_i].x - centroids[class_i].x;
			t.y = data[data_i].y - centroids[class_i].y;

			float dist = t.x * t.x + t.y * t.y;
	
			if (dist < min_dist) {
				partitioned[data_i] = class_i;
				min_dist = dist;
			}
		}
	}

	pthread_exit(NULL);
}

void *kmeans_t_update(void *thread_id)
{
	int id;
	int class_i;

	id = *((int *) thread_id);

	// Update step
	// Clear sum buffer and class count
	for (class_i = class_n*(id/NUM_THREADS); class_i < class_n*((id+1)/NUM_THREADS); class_i++) {
		centroids[class_i].x = 0.0;
		centroids[class_i].y = 0.0;
		count[class_i] = 0;
	}

	pthread_exit(NULL);
}

void *kmeans_t_sum(void *thread_id)
{
	int id;
	int data_i;

	id = *((int *) thread_id);

	// Sum up and count data for each class
	for (data_i = data_n*(id/NUM_THREADS); data_i < data_n*((id+1)/NUM_THREADS); data_i++) {
		centroids[partitioned[data_i]].x += data[data_i].x;
		centroids[partitioned[data_i]].y += data[data_i].y;
		count[partitioned[data_i]]++;
	}

	pthread_exit(NULL);
}

void *kmeans_t_divide(void *thread_id)
{
	int id;
	int class_i;

	id = *((int *) thread_id);

	// Divide the sum with number of class for mean point
	for (class_i = class_n*(id/NUM_THREADS); class_i < class_n*((id+1)/NUM_THREADS); class_i++) {
		centroids[class_i].x /= count[class_i];
		centroids[class_i].y /= count[class_i];
	}

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
			pthread_join(threads[t], NULL);
		}

		// Update //
		for(t=0; t<NUM_THREADS; t++)
		{
			retcode = pthread_create(&threads[t], NULL, 
				kmeans_t_update, (void*) t);

			printf("debug: %d\n", t);

			if(retcode)
			{
				printf("ERROR: return code from pthread_create() for id=%d is %d\n", t, retcode);
				exit(-1);
			}
		}
		for(t=0; t<NUM_THREADS; t++)
		{
			pthread_join(threads[t], NULL);
		}

		// Sum //
		for(t=0; t<NUM_THREADS; t++)
		{
			retcode = pthread_create(&threads[t], NULL, kmeans_t_sum, (void*) t);

			printf("debug: %d\n", t);

			if(retcode)
			{
				printf("ERROR: return code from pthread_create() for id=%d is %d\n", t, retcode);
				exit(-1);
			}
		}
		for(t=0; t<NUM_THREADS; t++)
		{
			pthread_join(threads[t], NULL);
		}

		// Divide //
		for(t=0; t<NUM_THREADS; t++)
		{
			retcode = pthread_create(&threads[t], NULL, kmeans_t_divide, (void*) t);

			printf("debug: %d\n", t);

			if(retcode)
			{
				printf("ERROR: return code from pthread_create() for id=%d is %d\n", t, retcode);
				exit(-1);
			}
		}
		for(t=0; t<NUM_THREADS; t++)
		{
			pthread_join(threads[t], NULL);
		}
	}
}

