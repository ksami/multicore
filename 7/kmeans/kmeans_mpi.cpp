#include "mpi.h"
#include "kmeans.h"

#include <stdlib.h>
#include <float.h>


void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
    // Loop indices for iteration, data and class
    int i, data_i, class_i;
    // Count number of data in each class
    int* count = (int*)malloc(sizeof(int) * class_n);
    // Temporal point value to calculate distance
    Point t;

    int numprocs, myid;

    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);


    // Iterate through number of interations
    for (i = 0; i < iteration_n; i++) {

        // Assignment step
        for (data_i = myid*data_n/numprocs; data_i < (myid+1)*data_n/numprocs; data_i++) {
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
    
        MPI_Barrier(MPI_COMM_WORLD);

        // Update step
        // Clear sum buffer and class count
        for (class_i = 0; class_i < class_n; class_i++) {
            centroids[class_i].x = 0.0;
            centroids[class_i].y = 0.0;
            count[class_i] = 0;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Sum up and count data for each class
        for (data_i = myid*data_n/numprocs; data_i < (myid+1)*data_n/numprocs; data_i++) {
            centroids[partitioned[data_i]].x += data[data_i].x;
            centroids[partitioned[data_i]].y += data[data_i].y;
            count[partitioned[data_i]]++;
        }
    
        MPI_Barrier(MPI_COMM_WORLD);

        // Divide the sum with number of class for mean point
        for (class_i = 0; class_i < class_n; class_i++) {
            centroids[class_i].x /= count[class_i];
            centroids[class_i].y /= count[class_i];
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

