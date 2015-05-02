#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include "timers.h"

#define NUM_THREADS 4
#define NDIM    2048
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

float a[NDIM][NDIM];
float b[NDIM][NDIM];
float c[NDIM][NDIM];

int print_matrix = 0;
int validation = 0;


void *mat_mul_t(void *thread_id)
{
	int id;
	int i, j, k;

	id = (int) thread_id;

	printf("Thread %d starting...", id);

	for( i = NDIM*(id/NUM_THREADS); i < NDIM*((id+1)/NUM_THREADS); i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			for( k = 0; k < NDIM; k++ )
			{
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	pthread_exit(NULL);
}

void mat_mul( float c[NDIM][NDIM], float a[NDIM][NDIM], float b[NDIM][NDIM] )
{
	pthread_t threads[NUM_THREADS];
	int retcode;
	int i;

	for(i=0; i<NUM_THREADS; i++)
	{
		retcode = pthread_create(&threads[i], NULL, mat_mul_t, (void *) i);

		if(retcode)
		{
			printf("ERROR: return code from pthread_create() for id=%d is %d\n", i, retcode);
			exit(-1);
		}
	}

	for(i=0; i<NUM_THREADS; i++)
	{
		pthread_join(threads[i], NULL);
	}

	pthread_exit(NULL);
	
	/* ijk */
	// for( i = 0; i < NDIM; i++ )
	// {
	// 	for( j = 0; j < NDIM; j++ )
	// 	{
	// 		for( k = 0; k < NDIM; k++ )
	// 		{
	// 			c[i][j] += a[i][k] * b[k][j];
	// 		}
	// 	}
	// }

}

/************************** DO NOT TOUCH BELOW HERE ******************************/

void check_mat_mul( float c[NDIM][NDIM], float a[NDIM][NDIM], float b[NDIM][NDIM] )
{
	int i, j, k;
	float sum;
	int validated = 1;

	printf("Validating the result..\n");
	
	// C = AB
	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			sum = 0;
			for( k = 0; k < NDIM; k++ )
			{
				sum += a[i][k] * b[k][j];
			}

			if( c[i][j] != sum )
			{
				printf("c[%d][%d] is differ(value=%lf correct_value=%lf)!!\n", i, j, c[i][j], sum );
				validated = 0;
			}
		}
	}

	printf("Validation : ");
	if( validated )
		printf("SUCCESSFUL.\n");
	else
		printf("FAILED.\n");
}

void print_mat( float mat[NDIM][NDIM] )
{
	int i, j;

	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			printf("%8.2lf ", mat[i][j]);
		}
		printf("\n");
	}
}

void print_help(const char* prog_name)
{
	printf("Usage: %s [-pvh]\n", prog_name );
	printf("\n");
	printf("OPTIONS\n");
	printf("  -p : print matrix data.\n");
	printf("  -v : validate matrix multiplication.\n");
	printf("  -h : print this page.\n");
}

void parse_opt(int argc, char** argv)
{
	int opt;

	while( (opt = getopt(argc, argv, "pvhikjs:")) != -1 )
	{
		switch(opt)
		{
		case 'p':
			// print matrix data.
			print_matrix = 1;
			break;

		case 'v':
			// validation
			validation = 1;
			break;

		case 'h':
		default:
			print_help(argv[0]);
			exit(0);
			break;
		}
	}
}

int main(int argc, char** argv)
{
	int i, j, k = 1;

	parse_opt( argc, argv );

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
	mat_mul( c, a, b );
	timer_stop(1);

	printf("Time elapsed : %lf sec\n", timer_read(1));


	if( validation )
		check_mat_mul( c, a, b );

	if( print_matrix )
	{
		printf("MATRIX A: \n");
		print_mat(a);

		printf("MATRIX B: \n");
		print_mat(b);

		printf("MATRIX C: \n");
		print_mat(c);
	}

	return 0;
}
