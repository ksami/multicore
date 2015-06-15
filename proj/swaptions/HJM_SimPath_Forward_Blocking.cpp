#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "HJM_type.h"
#include "HJM.h"
#include "nr_routines.h"

#ifdef TBB_VERSION
#include <pthread.h>
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/cache_aligned_allocator.h"


#define PARALLEL_B_GRAINSIZE 8


struct ParallelB {
  __volatile__ int l;
   FTYPE **pdZ;
  FTYPE **randZ;
  int BLOCKSIZE;
  int iN;

  ParallelB(FTYPE **pdZ_, FTYPE **randZ_, int BLOCKSIZE_, int iN_)//:
  //    pdZ(pdZ_), randZ(randZ_), BLOCKSIZE(BLOCKSIZE_), iN(iN_)
  {
    pdZ = pdZ_;
    randZ = randZ_;
    BLOCKSIZE = BLOCKSIZE_;
    iN = iN_; 
    /*fprintf(stderr,"(Construction object) pdZ=0x%08x, randZ=0x%08x\n",
      pdZ, randZ);*/

  }
  void set_l(int l_){l = l_;}

  void operator()(const tbb::blocked_range<int> &range) const {
    int begin = range.begin();
    int end   = range.end();
    int b,j;
    /*fprintf(stderr,"B: Thread %d from %d to %d. l=%d pdZ=0x%08x, BLOCKSIZE=%d, iN=%d\n",
      pthread_self(), begin, end, l,(int)pdZ,BLOCKSIZE,iN); */

    for(b=begin; b!=end; b++) {
      for (j=1;j<=iN-1;++j){
        pdZ[l][BLOCKSIZE*j + b]= CumNormalInv(randZ[l][BLOCKSIZE*j + b]);  /* 18% of the total executition time */
        //fprintf(stderr,"%d (%d, %d): [%d][%d]=%e\n",pthread_self(), begin, end,  l,BLOCKSIZE*j+b,pdZ[l][BLOCKSIZE*j + b]);
      }
    }

  }

};

#endif // TBB_VERSION



#ifdef ENABLE_OPENCL
#include <CL/cl.h>
#endif  //ENABLE_OPENCL



void serialB(FTYPE **pdZ, FTYPE **randZ, int BLOCKSIZE, int iN, int iFactors)
{

  //TODO:  
  for(int l=0;l<=iFactors-1;++l){
    for(int b=0; b<BLOCKSIZE; b++){
      for (int j=1;j<=iN-1;++j){
        pdZ[l][BLOCKSIZE*j + b]= CumNormalInv(randZ[l][BLOCKSIZE*j + b]);  /* 18% of the total executition time */
      }
    }
  }
}

int HJM_SimPath_Forward_Blocking(FTYPE **ppdHJMPath,    //Matrix that stores generated HJM path (Output)
                 int iN,                    //Number of time-steps
                 int iFactors,          //Number of factors in the HJM framework
                 FTYPE dYears,          //Number of years
                 FTYPE *pdForward,      //t=0 Forward curve
                 FTYPE *pdTotalDrift,   //Vector containing total drift corrections for different maturities
                 FTYPE **ppdFactors,    //Factor volatilities
                 long *lRndSeed,            //Random number seed
                 int BLOCKSIZE)
{   
//This function computes and stores an HJM Path for given inputs

#ifdef ENABLE_OPENCL
    int GLOBAL_WORK_ITEMS = BLOCKSIZE * iFactors * iN;
    cl_int result;
    int output[GLOBAL_WORK_ITEMS];  //debug

    // OpenCL //
    
    // The kernel index space is one dimensional
    // Specify the number of total work-items in the index space
    size_t global[1] = { GLOBAL_WORK_ITEMS };
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

    size_t sizeOutput = GLOBAL_WORK_ITEMS * sizeof(int);
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

    for(int i=0; i<GLOBAL_WORK_ITEMS; i++)
    {
        printf("%d\n", output[i]);
    }

#endif  //ENABLE_OPENCL



    int iSuccess = 0;
    int i,j,l; //looping variables
    FTYPE **pdZ; //vector to store random normals
    FTYPE **randZ; //vector to store random normals
    FTYPE dTotalShock; //total shock by which the forward curve is hit at (t, T-t)
    FTYPE ddelt, sqrt_ddelt; //length of time steps 

    ddelt = (FTYPE)(dYears/iN);
    sqrt_ddelt = sqrt(ddelt);

    pdZ   = dmatrix(0, iFactors-1, 0, iN*BLOCKSIZE -1); //assigning memory
    randZ = dmatrix(0, iFactors-1, 0, iN*BLOCKSIZE -1); //assigning memory

    //TODO:
    // =====================================================
    // t=0 forward curve stored iN first row of ppdHJMPath
    // At time step 0: insert expected drift 
    // rest reset to 0
    for(int b=0; b<BLOCKSIZE; b++){
      for(j=0;j<=iN-1;j++){
        ppdHJMPath[0][BLOCKSIZE*j + b] = pdForward[j]; 

        for(i=1;i<=iN-1;++i)
          { ppdHJMPath[i][BLOCKSIZE*j + b]=0; } //initializing HJMPath to zero
      }
    }
    // -----------------------------------------------------
    
    //TODO: 
    // =====================================================
    // sequentially generating random numbers

    for(int b=0; b<BLOCKSIZE; b++){
      for(int s=0; s<1; s++){
        for (j=1;j<=iN-1;++j){
          for (l=0;l<=iFactors-1;++l){
            //compute random number in exact same sequence
            randZ[l][BLOCKSIZE*j + b + s] = RanUnif(lRndSeed);  /* 10% of the total executition time */
          }
        }
      }
    }

    // =====================================================
    // shocks to hit various factors for forward curve at t

#ifdef TBB_VERSION
    ParallelB B(pdZ, randZ, BLOCKSIZE, iN);
    for(l=0;l<=iFactors-1;++l){
      B.set_l(l);
      tbb::parallel_for(tbb::blocked_range<int>(0, BLOCKSIZE, PARALLEL_B_GRAINSIZE),B);
    }

#else
    /* 18% of the total executition time */
    serialB(pdZ, randZ, BLOCKSIZE, iN, iFactors);
#endif

    //TODO:
    // =====================================================
    // Generation of HJM Path1
    for(int b=0; b<BLOCKSIZE; b++){ // b is the blocks
      for (j=1;j<=iN-1;++j) {// j is the timestep
        
        for (l=0;l<=iN-(j+1);++l){ // l is the future steps
          dTotalShock = 0;
          
          for (i=0;i<=iFactors-1;++i){// i steps through the stochastic factors
            dTotalShock += ppdFactors[i][l]* pdZ[i][BLOCKSIZE*j + b];               
          }            

          ppdHJMPath[j][BLOCKSIZE*l+b] = ppdHJMPath[j-1][BLOCKSIZE*(l+1)+b]+ pdTotalDrift[l]*ddelt + sqrt_ddelt*dTotalShock;
          //as per formula
        }
      }
    } // end Blocks
    // -----------------------------------------------------

    free_dmatrix(pdZ, 0, iFactors -1, 0, iN*BLOCKSIZE -1);
    free_dmatrix(randZ, 0, iFactors -1, 0, iN*BLOCKSIZE -1);
    iSuccess = 1;
    return iSuccess;
}
    


