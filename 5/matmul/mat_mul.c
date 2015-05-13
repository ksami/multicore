#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>
#include "timers.h"

// #define NUM_THREADS 4
// #define NDIM    2048
// #define MIN(x, y) (((x) < (y)) ? (x) : (y))

// float a[NDIM][NDIM];
// float b[NDIM][NDIM];
// float c[NDIM][NDIM];

// int print_matrix = 0;
// int validation = 0;

#define SIZE 100000 * 100000
// Kernel source code
const char* kernel_src = 
"__kernel void vec_add(__global const float* A, " 
"__global const float* B, "
"__global float* C) {"
" int id = get_global_id(0);"
" C[id] = A[id] + B[id];"
"}";

// void mat_mul( float c[NDIM][NDIM], float a[NDIM][NDIM], float b[NDIM][NDIM] )
// {

// }

// /************************** DO NOT TOUCH BELOW HERE ******************************/

// void check_mat_mul( float c[NDIM][NDIM], float a[NDIM][NDIM], float b[NDIM][NDIM] )
// {
//     int i, j, k;
//     float sum;
//     int validated = 1;

//     printf("Validating the result..\n");
    
//     // C = AB
//     for( i = 0; i < NDIM; i++ )
//     {
//         for( j = 0; j < NDIM; j++ )
//         {
//             sum = 0;
//             for( k = 0; k < NDIM; k++ )
//             {
//                 sum += a[i][k] * b[k][j];
//             }

//             if( c[i][j] != sum )
//             {
//                 printf("c[%d][%d] is differ(value=%lf correct_value=%lf)!!\n", i, j, c[i][j], sum );
//                 validated = 0;
//             }
//         }
//     }

//     printf("Validation : ");
//     if( validated )
//         printf("SUCCESSFUL.\n");
//     else
//         printf("FAILED.\n");
// }

// void print_mat( float mat[NDIM][NDIM] )
// {
//     int i, j;

//     for( i = 0; i < NDIM; i++ )
//     {
//         for( j = 0; j < NDIM; j++ )
//         {
//             printf("%8.2lf ", mat[i][j]);
//         }
//         printf("\n");
//     }
// }

// void print_help(const char* prog_name)
// {
//     printf("Usage: %s [-pvh]\n", prog_name );
//     printf("\n");
//     printf("OPTIONS\n");
//     printf("  -p : print matrix data.\n");
//     printf("  -v : validate matrix multiplication.\n");
//     printf("  -h : print this page.\n");
// }

// void parse_opt(int argc, char** argv)
// {
//     int opt;

//     while( (opt = getopt(argc, argv, "pvhikjs:")) != -1 )
//     {
//         switch(opt)
//         {
//         case 'p':
//             // print matrix data.
//             print_matrix = 1;
//             break;

//         case 'v':
//             // validation
//             validation = 1;
//             break;

//         case 'h':
//         default:
//             print_help(argv[0]);
//             exit(0);
//             break;
//         }
//     }
// }

int main(int argc, char** argv)
{
    int i;

    timer_init();

    // Vector Initialization //
    float* hostA;
    float* hostB;
    float* hostC;
    size_t sizeA, sizeB, sizeC;

    sizeA = SIZE * sizeof(float);
    sizeB = SIZE * sizeof(float);
    sizeC = SIZE * sizeof(float);
    hostA = (float*) malloc(sizeA);
    hostB = (float*) malloc(sizeB);
    hostC = (float*) malloc(sizeC);

    for (i = 0; i < SIZE; i++) {
        hostA[i] = (float) i;
        hostB[i] = (float) i * 2;
    }


    // OpenCL //
    // Obtain a list of available OpenCL platforms
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // Obtain the list of available devices on the OpenCL platform
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create an OpenCL context on a GPU device
    cl_context context;
    context = clCreateContext(0, 1, &device, NULL, NULL, NULL);


    // Create a command queue and attach it to the compute device
    // (in-order queue)
    cl_command_queue command_queue;
    command_queue = clCreateCommandQueue(context, device, 0, NULL);

    // Allocate buffer memory objects
    cl_mem bufferA;
    cl_mem bufferB;
    cl_mem bufferC;
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeA, NULL, NULL);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeB, NULL, NULL);
    bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeC, NULL, NULL);


    // Create an OpenCL program object for the context 
    // and load the kernel source into the program object
    cl_program program;
    size_t kernel_src_len = strlen(kernel_src);
    program = clCreateProgramWithSource(context, 1, (const char**) &kernel_src, &kernel_src_len, NULL);

    // Build (compile and link) the program executable 
    // from the source or binary for the device
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create a kernel object from the program
    cl_kernel kernel;
    kernel = clCreateKernel(program, "vec_add", NULL);


    // Set the arguments of the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &bufferC);

    // Copy the input vectors to the corresponding buffers
    clEnqueueWriteBuffer(command_queue, bufferA, CL_FALSE, 0, sizeA, hostA, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, bufferB, CL_FALSE, 0, sizeB, hostB, 0, NULL, NULL);


    // The kernel index space is one dimensional
    // Specify the number of total work-items in the index space
    size_t global[1] = { SIZE };
    // Specify the number of total work-items in a work-group
    size_t local[1] = { 16 };

    timer_start(1);
    //debug
    printf("executing kernel\n");

    // Execute the kernel
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global, local, 0, NULL, NULL);

    //debug
    printf("kernel executed\n");

    // Wait until the kernel command completes 
    // (no need to wait because the command_queue is an in-order queue)
    // clFinish(command_queue)
    // Copy the result from bufferC to hostC
    clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, sizeC, hostC, 0, NULL, NULL);
    
    timer_stop(1);

    printf("Time elapsed : %lf sec\n", timer_read(1));

    // Print the result
    // for (i = 0; i < SIZE; i++) {
    //     printf("C[%d] = %f\n", i, hostC[i]);
    // }

    return 0;
}



//     int i, j, k = 1;

//     parse_opt( argc, argv );

//     for( i = 0; i < NDIM; i++ )
//     {
//         for( j = 0; j < NDIM; j++ )
//         {
//             a[i][j] = k;
//             b[i][j] = k;
//             k++;
//         }
//     }

//     timer_start(1);
//     mat_mul( c, a, b );
//     timer_stop(1);

//     printf("Time elapsed : %lf sec\n", timer_read(1));


//     if( validation )
//         check_mat_mul( c, a, b );

//     if( print_matrix )
//     {
//         printf("MATRIX A: \n");
//         print_mat(a);

//         printf("MATRIX B: \n");
//         print_mat(b);

//         printf("MATRIX C: \n");
//         print_mat(c);
//     }

//     return 0;
// }
