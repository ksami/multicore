
#include "kmeans.h"

#include <stdlib.h>
#include <float.h>
#include <CL/cl.h>
#include <string.h>
#include <stdio.h>




/*
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef __USE_GNU
#define __USE_GNU
#endif

#include <execinfo.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>

// This structure mirrors the one found in /usr/include/asm/ucontext.h //
typedef struct _sig_ucontext {
 unsigned long     uc_flags;
 struct ucontext   *uc_link;
 stack_t           uc_stack;
 struct sigcontext uc_mcontext;
 sigset_t          uc_sigmask;
} sig_ucontext_t;

void crit_err_hdlr(int sig_num, siginfo_t * info, void * ucontext)
{
 void *             array[50];
 void *             caller_address;
 char **            messages;
 int                size, i;
 sig_ucontext_t *   uc;

 uc = (sig_ucontext_t *)ucontext;

 // Get the address at the time the signal was raised //
#if defined(__i386__) // gcc specific
 caller_address = (void *) uc->uc_mcontext.eip; // EIP: x86 specific
#elif defined(__x86_64__) // gcc specific
 caller_address = (void *) uc->uc_mcontext.rip; // RIP: x86_64 specific
#else
#error Unsupported architecture. // TODO: Add support for other arch.
#endif

 fprintf(stderr, "signal %d (%s), address is %p from %p\n", 
  sig_num, strsignal(sig_num), info->si_addr, 
  (void *)caller_address);

 size = backtrace(array, 50);

 // overwrite sigaction with caller's address //
 array[1] = caller_address;

 messages = backtrace_symbols(array, size);

 // skip first stack frame (points here) //
 for (i = 1; i < size && messages != NULL; ++i)
 {
  fprintf(stderr, "[bt]: (%d) %s\n", i, messages[i]);
 }

 free(messages);

 exit(EXIT_FAILURE);
}
*/





// Kernel source code
const char* kernel_src =
"__kernel void assign("
"__global const int class_n,"
"__global const Point* data,"
"__global const Point* centroids,"
"__global int* partitioned) {"
"    Point t;"
"    int class_i;"
"    int data_i = get_global_id(0);"
"    float dist;"
"    float min_dist = DBL_MAX;"
""
"    for (class_i = 0; class_i < class_n; class_i++) {"
"        t.x = data[data_i].x - centroids[class_i].x;"
"        t.y = data[data_i].y - centroids[class_i].y;"
""
"        dist = t.x * t.x + t.y * t.y;"
""
"        if (dist < min_dist) {"
"            partitioned[data_i] = class_i;"
"            min_dist = dist;"
"        }"
"    }"
"}";


void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
    /*
    struct sigaction sigact;

    sigact.sa_sigaction = crit_err_hdlr;
    sigact.sa_flags = SA_RESTART | SA_SIGINFO;

    if (sigaction(SIGSEGV, &sigact, (struct sigaction *)NULL) != 0)
    {
     fprintf(stderr, "error setting signal handler for %d (%s)\n",
       SIGSEGV, strsignal(SIGSEGV));

     exit(EXIT_FAILURE);
    }
    */


    int x=0; //debug
    // OpenCL //
    
    // The kernel index space is one dimensional
    // Specify the number of total work-items in the index space
    size_t global[1] = { data_n };
    // Specify the number of total work-items in a work-group
    size_t local[1] = { 16 };


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
    cl_mem bufferData;
    cl_mem bufferCentroids;
    cl_mem bufferPartitioned;

    size_t sizeData, sizeCentroids, sizePartitioned;
    sizeData = data_n * sizeof(Point);
    sizeCentroids = class_n * sizeof(Point);
    sizePartitioned = data_n * sizeof(int);

    bufferData = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeData, NULL, NULL);
    bufferCentroids = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeCentroids, NULL, NULL);
    bufferPartitioned = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizePartitioned, NULL, NULL);


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
    kernel = clCreateKernel(program, "assign", NULL);
        

    // Set the arguments of the kernel
    clSetKernelArg(kernel, 0, sizeof(int), (void*) class_n);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &bufferData);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &bufferCentroids);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) &bufferPartitioned);
    
    // Copy the input vectors to the corresponding buffers
    clEnqueueWriteBuffer(command_queue, bufferData, CL_FALSE, 0, sizeData, data, 0, NULL, NULL);


    

    // Algorithm //

    
    // Loop indices for iteration, data and class
    int i, data_i, class_i;
    // Count number of data in each class
    int* count = (int*)malloc(sizeof(int) * class_n);


    // Iterate through number of interations
    for (i = 0; i < iteration_n; i++) {
        printf("iteration %d\n",i); //debug
        
        // Assignment step
        
        // Copy the input vectors to the corresponding buffers
        clEnqueueWriteBuffer(command_queue, bufferCentroids, CL_TRUE, 0, sizeCentroids, centroids, 0, NULL, NULL);
        printf("%d\n",x++); //debug

        // Execute the kernel
        clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
        printf("%d\n",x++); //debug

        // Wait until the kernel command completes 
        clFinish(command_queue);
        printf("%d\n",x++); //debug

        // Copy the result from bufferPartitioned to partitioned
        clEnqueueReadBuffer(command_queue, bufferPartitioned, CL_TRUE, 0, sizePartitioned, partitioned, 0, NULL, NULL);
        printf("%d\n",x++); //debug


        // Update step
        // Clear sum buffer and class count
        for (class_i = 0; class_i < class_n; class_i++) {
            centroids[class_i].x = 0.0;
            centroids[class_i].y = 0.0;
            count[class_i] = 0;
        }
        printf("%d\n",x++); //debug

        // Sum up and count data for each class
        for (data_i = 0; data_i < data_n; data_i++) {
            printf("%d,",data_i); //debug         
            centroids[partitioned[data_i]].x += data[data_i].x;
            printf("%d,",data_i); //debug         
            centroids[partitioned[data_i]].y += data[data_i].y;
            printf("%d,",data_i); //debug         
            count[partitioned[data_i]]++;
            printf("%d\n",data_i); //debug         
        }
        printf("%d\n",x++); //debug

        // Divide the sum with number of class for mean point
        for (class_i = 0; class_i < class_n; class_i++) {
            centroids[class_i].x /= count[class_i];
            centroids[class_i].y /= count[class_i];
        }
        printf("%d\n",x++); //debug

    }
}

