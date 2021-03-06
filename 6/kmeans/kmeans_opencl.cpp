
#include "kmeans.h"

#include <stdlib.h>
#include <float.h>
#include <CL/cl.h>
#include <string.h>
#include <stdio.h>


// Kernel source code
const char* kernel_src =
"typedef struct {"
"    int x;"
"    int y;"
"} Point;"
""
"__kernel void assign("
"__global const Point* data,"
"__global const Point* centroids,"
"__global int* partitioned) {"
"    Point t;"
"    int class_i;"
"    int class_n = 16;"
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

    cl_int result;

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
    result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create an OpenCL context on a GPU device
    cl_context context;
    context = clCreateContext(0, 1, &device, NULL, NULL, &result);

    // Create a command queue and attach it to the compute device
    // (in-order queue)
    cl_command_queue command_queue;
    command_queue = clCreateCommandQueue(context, device, 0, &result);


    // Allocate buffer memory objects
    cl_mem bufferData;
    cl_mem bufferCentroids;
    cl_mem bufferPartitioned;

    size_t sizeData, sizeCentroids, sizePartitioned;
    sizeData = data_n * sizeof(Point);
    sizeCentroids = class_n * sizeof(Point);
    sizePartitioned = data_n * sizeof(int);

    bufferData = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeData, NULL, &result);
    bufferCentroids = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeCentroids, NULL, &result);
    bufferPartitioned = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizePartitioned, NULL, &result);


    // Create an OpenCL program object for the context 
    // and load the kernel source into the program object
    cl_program program;
    size_t kernel_src_len = strlen(kernel_src);
    program = clCreateProgramWithSource(context, 1, (const char**) &kernel_src, &kernel_src_len, &result);
    if(result!=CL_SUCCESS) printf("prog err: %d\n",result);

    // Build (compile and link) the program executable 
    // from the source or binary for the device
    result = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (result != CL_SUCCESS) {
        char *buff_erro;
        cl_int errcode;
        size_t build_log_len;
        errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
        if (errcode) {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-1);
        }

       buff_erro = (char*) malloc(build_log_len);
        if (!buff_erro) {
            printf("malloc failed at line %d\n", __LINE__);
            exit(-2);
        }

        errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
        if (errcode) {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-3);
        }

        fprintf(stderr,"Build log: \n%s\n", buff_erro); //Be careful with  the fprint
        free(buff_erro);
        fprintf(stderr,"clBuildProgram failed\n");
        exit(EXIT_FAILURE);
    }

    // Create a kernel object from the program
    cl_kernel kernel;
    kernel = clCreateKernel(program, "assign", &result);
        

    // Set the arguments of the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &bufferData);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &bufferCentroids);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &bufferPartitioned);
    
    // Copy the input vectors to the corresponding buffers
    result = clEnqueueWriteBuffer(command_queue, bufferData, CL_FALSE, 0, sizeData, data, 0, NULL, NULL);


    

    // Algorithm //

    // Loop indices for iteration, data and class
    int i, data_i, class_i;
    // Count number of data in each class
    int* count = (int*)malloc(sizeof(int) * class_n);


    // Iterate through number of interations
    for (i = 0; i < iteration_n; i++) {
        // Assignment step
        
        // Copy the input vectors to the corresponding buffers
        result = clEnqueueWriteBuffer(command_queue, bufferCentroids, CL_TRUE, 0, sizeCentroids, centroids, 0, NULL, NULL);
        if(result != CL_SUCCESS) printf("err: %d",result);

        // Wait until the kernel command completes 
        result = clFinish(command_queue);
        if(result != CL_SUCCESS) printf("err: %d",result);

        // Execute the kernel
        result = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
        if(result != CL_SUCCESS) printf("err: %d",result);

        // Wait until the kernel command completes 
        result = clFinish(command_queue);
        if(result != CL_SUCCESS) printf("err: %d",result);

        // Copy the result from bufferPartitioned to partitioned
        result = clEnqueueReadBuffer(command_queue, bufferPartitioned, CL_TRUE, 0, sizePartitioned, partitioned, 0, NULL, NULL);
        if(result != CL_SUCCESS) printf("err: %d",result);

        // Wait until the kernel command completes 
        result = clFinish(command_queue);
        if(result != CL_SUCCESS) printf("err: %d",result);

        // Update step
        // Clear sum buffer and class count
        for (class_i = 0; class_i < class_n; class_i++) {
            centroids[class_i].x = 0.0;
            centroids[class_i].y = 0.0;
            count[class_i] = 0;
        }

        // Sum up and count data for each class
        for (data_i = 0; data_i < data_n; data_i++) {
            centroids[partitioned[data_i]].x += data[data_i].x;
            centroids[partitioned[data_i]].y += data[data_i].y;
            count[partitioned[data_i]]++;
        }

        // Divide the sum with number of class for mean point
        for (class_i = 0; class_i < class_n; class_i++) {
            centroids[class_i].x /= count[class_i];
            centroids[class_i].y /= count[class_i];
        }
    }
}

