//HJM_Securities.cpp
//Routines to compute various security prices using HJM framework (via Simulation).
//Authors: Mark Broadie, Jatin Dewanwala
//Collaborator: Mikhail Smelyanskiy, Jike Chong, Intel

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "nr_routines.h"
#include "HJM.h"
#include "HJM_Securities.h"
#include "HJM_type.h"

#ifdef ENABLE_OPENCL
#include <CL/cl.h>
#endif

#ifdef ENABLE_THREADS
#include <pthread.h>
#define MAX_THREAD 1024

#ifdef TBB_VERSION
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/cache_aligned_allocator.h"
tbb::cache_aligned_allocator<FTYPE> memory_ftype;
tbb::cache_aligned_allocator<parm> memory_parm;
#define TBB_GRAINSIZE 1
#endif // TBB_VERSION
#endif //ENABLE_THREADS


#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

int NUM_TRIALS = DEFAULT_NUM_TRIALS;
int nThreads = 1;
int nSwaptions = 1;
int iN = 11; 
FTYPE dYears = 5.5; 
int iFactors = 3; 
parm *swaptions;

// =================================================
FTYPE *dSumSimSwaptionPrice_global_ptr;
FTYPE *dSumSquareSimSwaptionPrice_global_ptr;
int chunksize;


#ifdef TBB_VERSION
struct Worker {
    Worker(){}
    void operator()(const tbb::blocked_range<int> &range) const {
        FTYPE pdSwaptionPrice[2];
        int begin = range.begin();
        int end   = range.end();

        for(int i=begin; i!=end; i++) {
            int iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike, 
                swaptions[i].dCompounding, swaptions[i].dMaturity, 
                swaptions[i].dTenor, swaptions[i].dPaymentInterval,
                swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears, 
                swaptions[i].pdYield, swaptions[i].ppdFactors,
                100, NUM_TRIALS, BLOCK_SIZE, 0);
            assert(iSuccess == 1);
            swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
            swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];

        }
    }
};

#endif //TBB_VERSION


void * worker(void *arg){
    int tid = *((int *)arg);
    FTYPE pdSwaptionPrice[2];

    int chunksize = nSwaptions/nThreads;
    int beg = tid*chunksize;
    int end = (tid+1)*chunksize;
    if(tid == nThreads -1 )
        end = nSwaptions;

    for(int i=beg; i < end; i++) {
        int iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike, 
            swaptions[i].dCompounding, swaptions[i].dMaturity, 
            swaptions[i].dTenor, swaptions[i].dPaymentInterval,
            swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears, 
            swaptions[i].pdYield, swaptions[i].ppdFactors,
            100, NUM_TRIALS, BLOCK_SIZE, 0);
        assert(iSuccess == 1);
        swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
        swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];
    }

    return NULL;    
}




//Please note: Whenever we type-cast to (int), we add 0.5 to ensure that the value is rounded to the correct number. 
//For instance, if X/Y = 0.999 then (int) (X/Y) will equal 0 and not 1 (as (int) rounds down).
//Adding 0.5 ensures that this does not happen. Therefore we use (int) (X/Y + 0.5); instead of (int) (X/Y);

int main(int argc, char *argv[])
{
    int iSuccess = 0;
    int i,j;
    
    FTYPE **factors=NULL;

#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
    printf("PARSEC Benchmark Suite Version "__PARSEC_XSTRING(PARSEC_VERSION)"\n"); 
    fflush(NULL);
#else
    printf("PARSEC Benchmark Suite\n");
    fflush(NULL);
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_begin(__parsec_swaptions);
#endif

    if(argc == 1)
    {
        fprintf(stderr," usage: \n\t-ns [number of swaptions (should be > number of threads]\n\t-sm [number of simulations]\n\t-nt [number of threads]\n"); 
        exit(1);
    }

    for (int j=1; j<argc; j++) {
        if (!strcmp("-sm", argv[j])) {NUM_TRIALS = atoi(argv[++j]);}
        else if (!strcmp("-nt", argv[j])) {nThreads = atoi(argv[++j]);} 
        else if (!strcmp("-ns", argv[j])) {nSwaptions = atoi(argv[++j]);} 
        else {
            fprintf(stderr," usage: \n\t-ns [number of swaptions (should be > number of threads]\n\t-sm [number of simulations]\n\t-nt [number of threads]\n"); 
        }
    }

    if(nSwaptions < nThreads) {
        nSwaptions = nThreads; 
    }

    printf("Number of Simulations: %d,  Number of threads: %d Number of swaptions: %d\n", NUM_TRIALS, nThreads, nSwaptions);

#ifdef ENABLE_THREADS

#ifdef TBB_VERSION
    tbb::task_scheduler_init init(nThreads);
#else
    pthread_t      *threads;
    pthread_attr_t  pthread_custom_attr;

    if ((nThreads < 1) || (nThreads > MAX_THREAD))
    {
        fprintf(stderr,"Number of threads must be between 1 and %d.\n", MAX_THREAD);
        exit(1);
    }
    threads = (pthread_t *) malloc(nThreads * sizeof(pthread_t));
    pthread_attr_init(&pthread_custom_attr);

#endif // TBB_VERSION

    if ((nThreads < 1) || (nThreads > MAX_THREAD))
    {
      fprintf(stderr,"Number of threads must be between 1 and %d.\n", MAX_THREAD);
      exit(1);
    }

#else
    if (nThreads != 1)
    {
      fprintf(stderr,"Number of threads must be 1 (serial version)\n");
      exit(1);
    }
#endif //ENABLE_THREADS


#ifdef ENABLE_OPENCL
    cl_int result;
    int output[nSwaptions];

    // OpenCL //
    
    // The kernel index space is one dimensional
    // Specify the number of total work-items in the index space
    size_t global[1] = { nSwaptions };
    // Specify the number of total work-items in a work-group
    size_t local[1] = { 16 };

    // Obtain a list of available OpenCL platforms
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // Obtain the list of available devices on the OpenCL platform
    cl_device_id device;
#ifdef OPENCL_CPU
    result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
#else
    result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
#endif
    if(result!=CL_SUCCESS) printOpenCLError("clGetDeviceIDs", result);

    // Create an OpenCL context on a GPU device
    cl_context context;
    context = clCreateContext(0, 1, &device, NULL, NULL, &result);
    if(result!=CL_SUCCESS) printOpenCLError("clCreateContext", result);

    // Create a command queue and attach it to the compute device
    // (in-order queue)
    cl_command_queue command_queue;
    command_queue = clCreateCommandQueue(context, device, 0, &result);
    if(result!=CL_SUCCESS) printOpenCLError("clCreateCommandQueue", result);


    // Allocate buffer memory objects
    cl_mem bufferOutput;
    // cl_mem bufferData;
    // cl_mem bufferCentroids;
    // cl_mem bufferPartitioned;

    size_t sizeOutput = nSwaptions * 16 * sizeof(int);
    // size_t sizeData, sizeCentroids, sizePartitioned;
    // sizeData = data_n * sizeof(Point);
    // sizeCentroids = class_n * sizeof(Point);
    // sizePartitioned = data_n * sizeof(int);

    bufferOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeOutput, NULL, &result);
    if(result!=CL_SUCCESS) printOpenCLError("clCreateBuffer", result);
    // bufferData = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeData, NULL, &result);
    // bufferCentroids = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeCentroids, NULL, &result);
    // bufferPartitioned = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizePartitioned, NULL, &result);


    // Create an OpenCL program object for the context 
    // and load the kernel source into the program object
    cl_program program;
    size_t program_src_len = strlen(program_src);
    program = clCreateProgramWithSource(context, 1, (const char**) &program_src, &program_src_len, &result);
    if(result!=CL_SUCCESS) printOpenCLError("clCreateProgramWithSource", result);

    // Build (compile and link) the program executable 
    // from the source or binary for the device
    result = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if(result!=CL_SUCCESS) printOpenCLError("clBuildProgram", result);
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
    kernel = clCreateKernel(program, kernel_name, &result);
    if(result!=CL_SUCCESS) printOpenCLError("clCreateKernel", result);
        

    // Set the arguments of the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &bufferOutput);
    // clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &bufferData);
    // clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &bufferCentroids);
    // clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &bufferPartitioned);
    
    // Copy the input vectors to the corresponding buffers
    // result = clEnqueueWriteBuffer(command_queue, bufferOutput, CL_FALSE, 0, sizeOutput, output, 0, NULL, NULL);
    // if(result!=CL_SUCCESS) printOpenCLError("clEnqueueWriteBuffer", result);
    
    // Execute the kernel
    result = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    if(result!=CL_SUCCESS) printOpenCLError("clEnqueueNDRangeKernel", result);
    
    // Copy the result from bufferOutput to output
    result = clEnqueueReadBuffer(command_queue, bufferOutput, CL_TRUE, 0, sizeOutput, output, 0, NULL, NULL);
    if(result != CL_SUCCESS) printOpenCLError("clEnqueueReadBuffer", result);

    result = clFinish(command_queue);
    if(result != CL_SUCCESS) printOpenCLError("clFinish", result);

    for(int i=0; i<nSwaptions; i++)
    {
        printf("%d\n", output[i]);
    }

#endif  //ENABLE_OPENCL


    // initialize input dataset
    factors = dmatrix(0, iFactors-1, 0, iN-2);
    //the three rows store vol data for the three factors
    factors[0][0]= .01;
    factors[0][1]= .01;
    factors[0][2]= .01;
    factors[0][3]= .01;
    factors[0][4]= .01;
    factors[0][5]= .01;
    factors[0][6]= .01;
    factors[0][7]= .01;
    factors[0][8]= .01;
    factors[0][9]= .01;

    factors[1][0]= .009048;
    factors[1][1]= .008187;
    factors[1][2]= .007408;
    factors[1][3]= .006703;
    factors[1][4]= .006065;
    factors[1][5]= .005488;
    factors[1][6]= .004966;
    factors[1][7]= .004493;
    factors[1][8]= .004066;
    factors[1][9]= .003679;

    factors[2][0]= .001000;
    factors[2][1]= .000750;
    factors[2][2]= .000500;
    factors[2][3]= .000250;
    factors[2][4]= .000000;
    factors[2][5]= -.000250;
    factors[2][6]= -.000500;
    factors[2][7]= -.000750;
    factors[2][8]= -.001000;
    factors[2][9]= -.001250;

    // setting up multiple swaptions
    swaptions = 
#ifdef TBB_VERSION
    (parm *)memory_parm.allocate(sizeof(parm)*nSwaptions, NULL);
#else
    (parm *)malloc(sizeof(parm)*nSwaptions);
#endif

    int k;
    for (i = 0; i < nSwaptions; i++) {
        swaptions[i].Id = i;
        swaptions[i].iN = iN;
        swaptions[i].iFactors = iFactors;
        swaptions[i].dYears = dYears;

        swaptions[i].dStrike =  (double)i / (double)nSwaptions; 
        swaptions[i].dCompounding =  0;
        swaptions[i].dMaturity =  1;
        swaptions[i].dTenor =  2.0;
        swaptions[i].dPaymentInterval =  1.0;

        swaptions[i].pdYield = dvector(0,iN-1);;
        swaptions[i].pdYield[0] = .1;
        for(j=1;j<=swaptions[i].iN-1;++j)
            swaptions[i].pdYield[j] = swaptions[i].pdYield[j-1]+.005;

        swaptions[i].ppdFactors = dmatrix(0, swaptions[i].iFactors-1, 0, swaptions[i].iN-2);
        for(k=0;k<=swaptions[i].iFactors-1;++k)
            for(j=0;j<=swaptions[i].iN-2;++j)
                swaptions[i].ppdFactors[k][j] = factors[k][j];
        }


    // **********Calling the Swaption Pricing Routine*****************
#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_begin();
#endif

#ifdef ENABLE_THREADS

#ifdef TBB_VERSION
    Worker w;
    tbb::parallel_for(tbb::blocked_range<int>(0,nSwaptions,TBB_GRAINSIZE),w);
#else

    int threadIDs[nThreads];
    for (i = 0; i < nThreads; i++) {
        threadIDs[i] = i;
        pthread_create(&threads[i], &pthread_custom_attr, worker, &threadIDs[i]);
    }
    for (i = 0; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);

#endif // TBB_VERSION   

#else
    int threadID=0;
    worker(&threadID);
#endif //ENABLE_THREADS

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_end();
#endif

    for (i = 0; i < nSwaptions; i++) {
        fprintf(stderr,"Swaption%d: [SwaptionPrice: %.10lf StdError: %.10lf] \n", 
            i, swaptions[i].dSimSwaptionMeanPrice, swaptions[i].dSimSwaptionStdError);

    }

    for (i = 0; i < nSwaptions; i++) {
        free_dvector(swaptions[i].pdYield, 0, swaptions[i].iN-1);
        free_dmatrix(swaptions[i].ppdFactors, 0, swaptions[i].iFactors-1, 0, swaptions[i].iN-2);
    }


#ifdef TBB_VERSION
    memory_parm.deallocate(swaptions, sizeof(parm));
#else
    free(swaptions);
#endif // TBB_VERSION

    //***********************************************************

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_end();
#endif

    return iSuccess;
}
