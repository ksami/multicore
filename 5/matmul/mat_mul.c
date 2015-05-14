#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>
#include "timers.h"

#define SIZE 10000
// Kernel source code
const char* kernel_src = 
"__kernel void mat_mul("
"__global const float** A, " 
"__global const float** B, "
"__global float** C,"
"__gloabl const int size) {"
"  int k;"
"  int id = get_global_id(0);"
"  int i = id/size;"
"  int j = id%size;"
"  for( k = 0; k < size; k++ ) {"
"    C[i][j] += A[i][k] * B[k][j];"
"  }"
"}";


int main(int argc, char** argv)
{
    int i, j, k = 1;

    timer_init();

    // Vector Initialization //
    float** hostA;
    float** hostB;
    float** hostC;
    size_t sizeA, sizeB, sizeC;

    sizeA = SIZE * SIZE * sizeof(float);
    sizeB = SIZE * SIZE * sizeof(float);
    sizeC = SIZE * SIZE * sizeof(float);
    hostA = (float**) malloc(SIZE * sizeof(float*));
    hostB = (float**) malloc(SIZE * sizeof(float*));
    hostC = (float**) malloc(SIZE * sizeof(float*));

    for(i=0; i<SIZE; i++)
    {
      hostA[i] = (float*) malloc(SIZE * sizeof(float));
      hostB[i] = (float*) malloc(SIZE * sizeof(float));
      hostC[i] = (float*) malloc(SIZE * sizeof(float));
    }

    for( i = 0; i < SIZE; i++ )
    {
        for( j = 0; j < SIZE; j++ )
        {
            hostA[i][j] = k;
            hostB[i][j] = k;
            k++;
        }
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
    kernel = clCreateKernel(program, "mat_mul", NULL);


    // Set the arguments of the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), (void*) SIZE);


    // Copy the input vectors to the corresponding buffers
    clEnqueueWriteBuffer(command_queue, bufferA, CL_FALSE, 0, sizeA, hostA, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, bufferB, CL_FALSE, 0, sizeB, hostB, 0, NULL, NULL);


    // The kernel index space is one dimensional
    // Specify the number of total work-items in the index space
    size_t global[1] = { SIZE*SIZE };
    // Specify the number of total work-items in a work-group
    size_t local[1] = { 8 };


    timer_start(1);

    // Execute the kernel
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global, local, 0, NULL, NULL);


    // Wait until the kernel command completes 
    // (no need to wait because the command_queue is an in-order queue)
    clFinish(command_queue);
    timer_stop(1);

    // Copy the result from bufferC to hostC
    clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, sizeC, hostC, 0, NULL, NULL);
    


    printf("Time elapsed : %lf sec\n", timer_read(1));

    // Print the result
    // for (i = 0; i < SIZE; i++) {
    //     printf("C[%d] = %f\n", i, hostC[i]);
    // }
    return 0;
}
