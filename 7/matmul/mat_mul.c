#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "timers.h"

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
    int numprocs, myid;
    int i, j, k=1;
    double start, end;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    if(myid == 0)
    {
        for( i = 0; i < NDIM; i++ )
        {
            for( j = 0; j < NDIM; j++ )
            {
                a[i][j] = k;
                b[i][j] = k;
                c[i][j] = 0;
                k++;
            }
        }
    }

    MPI_Bcast(a, NDIM*NDIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, NDIM*NDIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(c, NDIM*NDIM, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(myid==0) 
        start = get_time();

    for( i = myid; i < NDIM; i+=numprocs )
    {
        for( j = 0; j < NDIM; j++ )
        {
            for( k = 0; k < NDIM; k++ )
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(myid == 0)
    {
        end = get_time();
        printf("Time elapsed : %lf sec\n", end-start);
    }

    MPI_Finalize();
    return 0;
}
