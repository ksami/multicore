
#include "kmeans.h"

#include <stdlib.h>
#include <float.h>

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

    







    
    // Loop indices for iteration, data and class
    int i, data_i, class_i;
    // Count number of data in each class
    int* count = (int*)malloc(sizeof(int) * class_n);
    // Temporal point value to calculate distance
    Point t;


    // Iterate through number of interations
    for (i = 0; i < iteration_n; i++) {

        // Assignment step
        for (data_i = 0; data_i < data_n; data_i++) {
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

