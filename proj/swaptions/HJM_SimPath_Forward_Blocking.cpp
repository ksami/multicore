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

int done_serialB=0;


#ifdef ENABLE_OPENCL
#include <CL/cl.h>
#define MAX_DEV 16
int ndev = 1;

const char* name_init = "op_init";
const char* name_serialB = "op_serialB";
const char* name_randGen = "op_randGen";
const char* name_pathGen = "op_pathGen";
const char* program_src =
"#define FTYPE double\n"
"\n"
"/**********************************************************************/\n"
"static FTYPE a[4] = {\n"
"  2.50662823884,\n"
"    -18.61500062529,\n"
"    41.39119773534,\n"
"    -25.44106049637\n"
"};\n"
"\n"
"static FTYPE b[4] = {\n"
"  -8.47351093090,\n"
"    23.08336743743,\n"
"    -21.06224101826,\n"
"    3.13082909833\n"
"};\n"
"\n"
"static FTYPE c[9] = {\n"
"  0.3374754822726147,\n"
"    0.9761690190917186,\n"
"    0.1607979714918209,\n"
"    0.0276438810333863,\n"
"    0.0038405729373609,\n"
"    0.0003951896511919,\n"
"    0.0000321767881768,\n"
"    0.0000002888167364,\n"
"    0.0000003960315187\n"
"};\n"
"\n"
"/**********************************************************************/\n"
"FTYPE CumNormalInv( FTYPE u )\n"
"{\n"
"  \n"
"  FTYPE x, r;\n"
"  \n"
"  x = u - 0.5;\n"
"  if( fabs (x) < 0.42 )\n"
"  { \n"
"    r = x * x;\n"
"    r = x * ((( a[3]*r + a[2]) * r + a[1]) * r + a[0])/\n"
"          ((((b[3] * r+ b[2]) * r + b[1]) * r + b[0]) * r + 1.0);\n"
"    return (r);\n"
"  }\n"
"  \n"
"  r = u;\n"
"  if( x > 0.0 ) r = 1.0 - u;\n"
"  r = log(-log(r));\n"
"  r = c[0] + r * (c[1] + r * \n"
"       (c[2] + r * (c[3] + r * \n"
"       (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r*c[8])))))));\n"
"  if( x < 0.0 ) r = -r;\n"
"  \n"
"  return (r);\n"
"  \n"
"} // end of CumNormalInv\n"
"\n"
"__kernel void op_serialB(__global FTYPE *pdZ, __global FTYPE *randZ, __global FTYPE* output, __global int* input)\n"
"{\n"
"  int BLOCKSIZE = input[0];\n"
"  int iFactors = input[1];\n"
"  int iN = input[2];\n"
"  int rowsize = BLOCKSIZE * iN;\n"
"  int id = get_global_id(0);\n"
"\n"
"  for(int i=id*iFactors;i<(id+1)*iFactors;i++){\n"
"        pdZ[i]= CumNormalInv(randZ[i]);  \n"
"        output[id] = randZ[i];\n"
"  }\n"
"}\n"
"\n"
"FTYPE RanUnif( int s )\n"
"{\n"
"  // uniform random number generator\n"
"  long   ix, k1;\n"
"  FTYPE dRes;\n"
"  \n"
"  ix = s;\n"
"  ix *= 1513517L;\n"
"  ix %= 2147483647L; \n"
"  k1 = ix/127773L;\n"
"  ix = 16807L*( ix - k1*127773L ) - k1 * 2836L;\n"
"  if (ix < 0) ix = ix + 2147483647L;\n"
"  dRes = (ix * 4.656612875e-10);\n"
"  return (dRes);\n"
"  \n"
"} // end of RanUnif\n"
"\n"
"__kernel void op_randGen(__global FTYPE* randZ, __global FTYPE* output, __global int* input)\n"
"{\n"
"  int BLOCKSIZE = input[0];\n"
"  int iFactors = input[1];\n"
"  int iN = input[2];\n"
"  int id = get_global_id(0);\n"
"  for(int i=id*iFactors;i<(id+1)*iFactors;i++){\n"
"        randZ[i]= RanUnif(input[3]+i);  \n"
"        output[id] = randZ[i];\n"
"  }\n"
"}\n"
"\n"
"__kernel void op_init(__global FTYPE* ppdHJMPath, __global FTYPE* pdForward, __global FTYPE* output, __global int* input)\n"
"{\n"
"  int BLOCKSIZE = input[0];\n"
"  int iFactors = input[1];\n"
"  int iN = input[2];\n"
"  int id = get_global_id(0);\n"
"\n"
"  ppdHJMPath[id] = pdForward[id/BLOCKSIZE];\n"
"  for(int i=1;i<iN;i++)\n"
"    { ppdHJMPath[(i*BLOCKSIZE*iN) + id]=0; } //initializing HJMPath to zero\n"
"}\n"
"\n"
"__kernel void op_pathGen(__global FTYPE* ppdHJMPath, __global FTYPE* ppdFactors, __global FTYPE* pdZ, __global FTYPE* pdTotalDrift, __global FTYPE* output, __global int* input)\n"
"{\n"
"  int BLOCKSIZE = input[0];\n"
"  int iFactors = input[1];\n"
"  int iN = input[2];\n"
"  int id = get_global_id(0);\n"
"  FTYPE dTotalShock = 0;\n"
"  FTYPE ddelt = (FTYPE)(5.5/iN);\n"
"  FTYPE sqrt_ddelt = sqrt(ddelt);\n"
"\n"
"  if(id<BLOCKSIZE){\n"
"    int b=id;\n"
"    for (int j=1;j<=iN-1;++j) {// j is the timestep\n"
"      for (int l=0;l<=iN-(j+1);++l){ // l is the future steps\n"
"        dTotalShock = 0;\n"
"        for (int i=0;i<=iFactors-1;++i){// i steps through the stochastic factors\n"
"          dTotalShock += ppdFactors[(i*(iN-1))+l]* pdZ[(i*BLOCKSIZE*iN) + BLOCKSIZE*j + b];\n"
"        }            \n"
"  \n"
"        ppdHJMPath[(j*BLOCKSIZE*iN)+BLOCKSIZE*l+b] = ppdHJMPath[((j-1)*BLOCKSIZE*iN)+BLOCKSIZE*(l+1)+b]+ pdTotalDrift[l]*ddelt + sqrt_ddelt*dTotalShock;\n"
"      }\n"
"    }\n"
"  }\n"
"}\n";

cl_platform_id platform;
cl_device_id device[MAX_DEV];
cl_context context;
cl_command_queue command_queue[MAX_DEV];
cl_program program;
cl_kernel kernel_init[MAX_DEV];
cl_kernel kernel_serialB[MAX_DEV];
cl_kernel kernel_randGen[MAX_DEV];
cl_kernel kernel_pathGen[MAX_DEV];
size_t sizeOutput;
size_t sizeInput;
size_t sizepdZ;
size_t sizerandZ;
size_t sizeppdHJMPath;
size_t sizepdForward;
size_t sizeppdFactors;
size_t sizepdTotalDrift;
cl_mem bufferOutput[MAX_DEV];
cl_mem bufferInput[MAX_DEV];
cl_mem bufferpdZ[MAX_DEV];
cl_mem bufferrandZ[MAX_DEV];
cl_mem bufferppdHJMPath[MAX_DEV];
cl_mem bufferpdForward[MAX_DEV];
cl_mem bufferppdFactors[MAX_DEV];
cl_mem bufferpdTotalDrift[MAX_DEV];

// OpenCL Errors //
void printOpenCLError(char* functionName, cl_int error)
{
    printf(functionName);
    printf(": ");
    switch(error){
        // run-time and JIT compiler errors
        case 0: printf("CL_SUCCESS"); break;
        case -1: printf("CL_DEVICE_NOT_FOUND"); break;
        case -2: printf("CL_DEVICE_NOT_AVAILABLE"); break;
        case -3: printf("CL_COMPILER_NOT_AVAILABLE"); break;
        case -4: printf("CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
        case -5: printf("CL_OUT_OF_RESOURCES"); break;
        case -6: printf("CL_OUT_OF_HOST_MEMORY"); break;
        case -7: printf("CL_PROFILING_INFO_NOT_AVAILABLE"); break;
        case -8: printf("CL_MEM_COPY_OVERLAP"); break;
        case -9: printf("CL_IMAGE_FORMAT_MISMATCH"); break;
        case -10: printf("CL_IMAGE_FORMAT_NOT_SUPPORTED"); break;
        case -11: printf("CL_BUILD_PROGRAM_FAILURE"); break;
        case -12: printf("CL_MAP_FAILURE"); break;
        case -13: printf("CL_MISALIGNED_SUB_BUFFER_OFFSET"); break;
        case -14: printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"); break;
        case -15: printf("CL_COMPILE_PROGRAM_FAILURE"); break;
        case -16: printf("CL_LINKER_NOT_AVAILABLE"); break;
        case -17: printf("CL_LINK_PROGRAM_FAILURE"); break;
        case -18: printf("CL_DEVICE_PARTITION_FAILED"); break;
        case -19: printf("CL_KERNEL_ARG_INFO_NOT_AVAILABLE"); break;

        // compile-time errors
        case -30: printf("CL_INVALID_VALUE"); break;
        case -31: printf("CL_INVALID_DEVICE_TYPE"); break;
        case -32: printf("CL_INVALID_PLATFORM"); break;
        case -33: printf("CL_INVALID_DEVICE"); break;
        case -34: printf("CL_INVALID_CONTEXT"); break;
        case -35: printf("CL_INVALID_QUEUE_PROPERTIES"); break;
        case -36: printf("CL_INVALID_COMMAND_QUEUE"); break;
        case -37: printf("CL_INVALID_HOST_PTR"); break;
        case -38: printf("CL_INVALID_MEM_OBJECT"); break;
        case -39: printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"); break;
        case -40: printf("CL_INVALID_IMAGE_SIZE"); break;
        case -41: printf("CL_INVALID_SAMPLER"); break;
        case -42: printf("CL_INVALID_BINARY"); break;
        case -43: printf("CL_INVALID_BUILD_OPTIONS"); break;
        case -44: printf("CL_INVALID_PROGRAM"); break;
        case -45: printf("CL_INVALID_PROGRAM_EXECUTABLE"); break;
        case -46: printf("CL_INVALID_KERNEL_NAME"); break;
        case -47: printf("CL_INVALID_KERNEL_DEFINITION"); break;
        case -48: printf("CL_INVALID_KERNEL"); break;
        case -49: printf("CL_INVALID_ARG_INDEX"); break;
        case -50: printf("CL_INVALID_ARG_VALUE"); break;
        case -51: printf("CL_INVALID_ARG_SIZE"); break;
        case -52: printf("CL_INVALID_KERNEL_ARGS"); break;
        case -53: printf("CL_INVALID_WORK_DIMENSION"); break;
        case -54: printf("CL_INVALID_WORK_GROUP_SIZE"); break;
        case -55: printf("CL_INVALID_WORK_ITEM_SIZE"); break;
        case -56: printf("CL_INVALID_GLOBAL_OFFSET"); break;
        case -57: printf("CL_INVALID_EVENT_WAIT_LIST"); break;
        case -58: printf("CL_INVALID_EVENT"); break;
        case -59: printf("CL_INVALID_OPERATION"); break;
        case -60: printf("CL_INVALID_GL_OBJECT"); break;
        case -61: printf("CL_INVALID_BUFFER_SIZE"); break;
        case -62: printf("CL_INVALID_MIP_LEVEL"); break;
        case -63: printf("CL_INVALID_GLOBAL_WORK_SIZE"); break;
        case -64: printf("CL_INVALID_PROPERTY"); break;
        case -65: printf("CL_INVALID_IMAGE_DESCRIPTOR"); break;
        case -66: printf("CL_INVALID_COMPILER_OPTIONS"); break;
        case -67: printf("CL_INVALID_LINKER_OPTIONS"); break;
        case -68: printf("CL_INVALID_DEVICE_PARTITION_COUNT"); break;

        // extension errors
        case -1000: printf("CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR"); break;
        case -1001: printf("CL_PLATFORM_NOT_FOUND_KHR"); break;
        case -1002: printf("CL_INVALID_D3D10_DEVICE_KHR"); break;
        case -1003: printf("CL_INVALID_D3D10_RESOURCE_KHR"); break;
        case -1004: printf("CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR"); break;
        case -1005: printf("CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR"); break;
        default: printf("Unknown OpenCL error"); break;
    }
    printf("\n");
}

#endif  //ENABLE_OPENCL



void serialB(FTYPE *pdZ, FTYPE *randZ, int BLOCKSIZE, int iN, int iFactors)
{
  for(int l=0;l<=iFactors-1;++l){
    for(int b=0; b<BLOCKSIZE; b++){
      for (int j=1;j<=iN-1;++j){
        pdZ[(l*BLOCKSIZE*iN) + BLOCKSIZE*j + b]= CumNormalInv(randZ[(l*BLOCKSIZE*iN) + BLOCKSIZE*j + b]);  /* 18% of the total executition time */
      }
    }
  }
}

int HJM_SimPath_Forward_Blocking(FTYPE *ppdHJMPath,    //Matrix that stores generated HJM path (Output)
                 int iN,                    //Number of time-steps
                 int iFactors,          //Number of factors in the HJM framework
                 FTYPE dYears,          //Number of years
                 FTYPE *pdForward,      //t=0 Forward curve
                 FTYPE *pdTotalDrift,   //Vector containing total drift corrections for different maturities
                 FTYPE **dppdFactors,    //Factor volatilities
                 long *lRndSeed,            //Random number seed
                 int BLOCKSIZE, FTYPE* pdZ, FTYPE* randZ)
{   
//This function computes and stores an HJM Path for given inputs
#ifdef ENABLE_OPENCL
    int GLOBAL_WORK_ITEMS = 176;
    cl_int result;
    FTYPE output[GLOBAL_WORK_ITEMS];  //debug
    int input[4] = {0, 0, 0, 0};  //debug

    for(int i=0; i<BLOCKSIZE*iN*iFactors; i++)
    {
        pdZ[i] = 0.0;
        randZ[i] = 0.0;
    }

    // OpenCL //
    
    // The kernel index space is one dimensional
    // Specify the number of total work-items in the index space
    size_t global[1] = { GLOBAL_WORK_ITEMS };
    // Specify the number of total work-items in a work-group
    size_t local[1] = { 16 };


    if(done_serialB==0){

    // Obtain a list of available OpenCL platforms
    //cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // Obtain the list of available devices on the OpenCL platform
    //cl_device_id device;
#ifdef OPENCL_CPU
    result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, (unsigned int*)&ndev);
    result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, ndev, device, NULL);
#else
    result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, (unsigned int*)&ndev);
    result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, ndev, device, NULL);
#endif
    if(result!=CL_SUCCESS) printOpenCLError("clGetDeviceIDs", result);

    // Create an OpenCL context on a GPU device
    //cl_context context;
    context = clCreateContext(0, ndev, device, NULL, NULL, &result);
    if(result!=CL_SUCCESS) printOpenCLError("clCreateContext", result);

    printf("context called, ndev: %d\n", ndev);
    
    // Create a command queue and attach it to the compute device
    // (in-order queue)
    //cl_command_queue command_queue;
    for (int i = 0; i < ndev; i++)
    {
        printf("%d queue\n",i);
        command_queue[i] = clCreateCommandQueue(context, device[i], 0, &result);
        printf("%d queue done\n",i);
        if(result!=CL_SUCCESS) printOpenCLError("clCreateCommandQueue", result);
    }
    printf("queues called\n");
    



    // Create an OpenCL program object for the context 
    // and load the kernel source into the program object

    size_t program_src_len = strlen(program_src);
    program = clCreateProgramWithSource(context, 1, (const char**) &program_src, &program_src_len, &result);
    if(result!=CL_SUCCESS) printOpenCLError("clCreateProgramWithSource", result);

    // Build (compile and link) the program executable 
    // from the source or binary for the device
    result = clBuildProgram(program, ndev, device, NULL, NULL, NULL);
    if(result!=CL_SUCCESS) printOpenCLError("clBuildProgram", result);
    if (result != CL_SUCCESS) {
        char *buff_erro;
        cl_int errcode;
        size_t build_log_len;
        errcode = clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
        if (errcode) {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-1);
        }

       buff_erro = (char*) malloc(build_log_len);
        if (!buff_erro) {
            printf("malloc failed at line %d\n", __LINE__);
            exit(-2);
        }

        errcode = clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
        if (errcode) {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-3);
        }

        fprintf(stderr,"Build log: \n%s\n", buff_erro); //Be careful with  the fprint
        free(buff_erro);
        fprintf(stderr,"clBuildProgram failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < ndev; i++) {
        // Create a kernel object from the program
        kernel_init[i] = clCreateKernel(program, name_init, &result);
        if(result!=CL_SUCCESS) printOpenCLError("clCreateKernel", result);
        kernel_serialB[i] = clCreateKernel(program, name_serialB, &result);
        if(result!=CL_SUCCESS) printOpenCLError("clCreateKernel", result);
        kernel_randGen[i] = clCreateKernel(program, name_randGen, &result);
        if(result!=CL_SUCCESS) printOpenCLError("clCreateKernel", result);
        kernel_pathGen[i] = clCreateKernel(program, name_pathGen, &result);
        if(result!=CL_SUCCESS) printOpenCLError("clCreateKernel", result);
    }

    // Allocate buffer memory objects        
    sizeOutput = GLOBAL_WORK_ITEMS * sizeof(FTYPE);
    sizeInput = 4 * sizeof(int);
    sizepdZ = iFactors * iN * BLOCKSIZE * sizeof(FTYPE);
    sizerandZ = iFactors * iN * BLOCKSIZE * sizeof(FTYPE);
    sizeppdHJMPath = iN * iN * BLOCKSIZE * sizeof(FTYPE);
    sizepdForward = iN * sizeof(FTYPE);
    sizeppdFactors = iFactors * (iN-1) * sizeof(FTYPE);
    sizepdTotalDrift = (iN-1) * sizeof(FTYPE);
    
    for (int i = 0; i < ndev; i++) 
    {   
        bufferOutput[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeOutput, NULL, &result);
        if(result!=CL_SUCCESS) printOpenCLError("clCreateBuffer", result);
        bufferInput[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeInput, NULL, &result);
        if(result!=CL_SUCCESS) printOpenCLError("clCreateBuffer", result);
        bufferpdZ[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizepdZ, NULL, &result);
        if(result!=CL_SUCCESS) printOpenCLError("clCreateBuffer", result);
        bufferrandZ[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizerandZ, NULL, &result);
        if(result!=CL_SUCCESS) printOpenCLError("clCreateBuffer", result);
        bufferppdHJMPath[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeppdHJMPath, NULL, &result);
        if(result!=CL_SUCCESS) printOpenCLError("clCreateBuffer", result);
        bufferpdForward[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizepdForward, NULL, &result);
        if(result!=CL_SUCCESS) printOpenCLError("clCreateBuffer", result);
        bufferppdFactors[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeppdFactors, NULL, &result);
        if(result!=CL_SUCCESS) printOpenCLError("clCreateBuffer", result);
        bufferpdTotalDrift[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizepdTotalDrift, NULL, &result);
        if(result!=CL_SUCCESS) printOpenCLError("clCreateBuffer", result);
    }
    done_serialB=1;


    }  //done_serialB

#endif  //ENABLE_OPENCL


    int iSuccess = 0;
    int i,j,l; //looping variables

    FTYPE dTotalShock; //total shock by which the forward curve is hit at (t, T-t)
    FTYPE ddelt, sqrt_ddelt; //length of time steps
    FTYPE *ppdFactors;
    ppdFactors = (FTYPE*) malloc(iFactors*(iN-1)*sizeof(FTYPE));
    for(int i=0; i<iFactors*(iN-1); i++)
      ppdFactors[i] = dppdFactors[i/(iN-1)][i%(iN-1)];

    ddelt = (FTYPE)(dYears/iN);
    sqrt_ddelt = sqrt(ddelt);


    // pdZ   = dmatrix(0, iFactors-1, 0, iN*BLOCKSIZE -1); //assigning memory
    // randZ = dmatrix(0, iFactors-1, 0, iN*BLOCKSIZE -1); //assigning memory


    // =====================================================
    // t=0 forward curve stored iN first row of ppdHJMPath
    // At time step 0: insert expected drift 
    // rest reset to 0
#ifdef ENABLE_OPENCL
    input[0] = BLOCKSIZE;
    input[1] = iFactors;
    input[2] = iN;
    input[3] = 100;

    for (int i = 0; i < ndev; i++)
    {
        clSetKernelArg(kernel_init[i], 0, sizeof(cl_mem), (void*) &bufferppdHJMPath[i]);
        clSetKernelArg(kernel_init[i], 1, sizeof(cl_mem), (void*) &bufferpdForward[i]);
        clSetKernelArg(kernel_init[i], 2, sizeof(cl_mem), (void*) &bufferOutput[i]);
        clSetKernelArg(kernel_init[i], 3, sizeof(cl_mem), (void*) &bufferInput[i]);
    }

    for (int i = 0; i < ndev; i++)
    {
        // Set input
        // TODO: stopped here, makeing opencl mult devicen
        result = clEnqueueWriteBuffer(command_queue[i], bufferInput[i], CL_FALSE, 0, sizeInput, input, 0, NULL, NULL);
        if(result!=CL_SUCCESS) printOpenCLError("clEnqueueWriteBuffer", result);
        result = clEnqueueWriteBuffer(command_queue[i], bufferpdForward[i], CL_FALSE, 0, sizepdForward, pdForward, 0, NULL, NULL);
        if(result!=CL_SUCCESS) printOpenCLError("clEnqueueWriteBuffer", result);
    }

    for (int i = 0; i < ndev; i++)
    {
        // Execute the kernel_init
        result = clEnqueueNDRangeKernel(command_queue[i], kernel_init[i], 1, NULL, global, local, 0, NULL, NULL);
        if(result!=CL_SUCCESS) printOpenCLError("clEnqueueNDRangeKernel", result);
    }
    
    for (int i = 0; i < ndev; i++)
    {
        // Wait for all commands in queue to finish
        result = clFinish(command_queue[i]);
        if(result != CL_SUCCESS) printOpenCLError("clFinish", result);
    }

    for (int i = 0; i < ndev; i++)
    {
        // Copy the result from bufferOutput to output
        // result = clEnqueueReadBuffer(command_queue, bufferOutput, CL_TRUE, 0, sizeOutput, output, 0, NULL, NULL);
        // if(result != CL_SUCCESS) printOpenCLError("clEnqueueReadBuffer", result);
        result = clEnqueueReadBuffer(command_queue[i], bufferppdHJMPath[i], CL_TRUE, 0, sizeppdHJMPath, ppdHJMPath, 0, NULL, NULL);
        if(result != CL_SUCCESS) printOpenCLError("clEnqueueReadBuffer", result);
    }

    // debug check output
    // for(int i=0; i<GLOBAL_WORK_ITEMS; i++)
    // {
    //    if(output[i]) printf("%d: %lf\n", i, output[i]);
    // }
#else
    for(int b=0; b<BLOCKSIZE; b++){
      for(j=0;j<=iN-1;j++){
        ppdHJMPath[BLOCKSIZE*j + b] = pdForward[j]; 

        for(i=1;i<=iN-1;++i)
          { ppdHJMPath[(i*iN*BLOCKSIZE)+BLOCKSIZE*j + b]=0; } //initializing HJMPath to zero
      }
    }
#endif  //ENABLE_OPENCL
    // -----------------------------------------------------
    

    // =====================================================
    // sequentially generating random numbers

    //for(int b=0; b<BLOCKSIZE; b++){
      //for(int s=0; s<1; s++){
        //for (j=1;j<=iN-1;++j){
          //for (l=0;l<=iFactors-1;++l){
            //compute random number in exact same sequence
            //randZ[(l*BLOCKSIZE*iN) + BLOCKSIZE*j + b + s] = RanUnif(lRndSeed);  /* 10% of the total executition time */
#ifdef ENABLE_OPENCL

    for (int i = 0; i < ndev; i++)
    {
        clSetKernelArg(kernel_randGen[i], 0, sizeof(cl_mem), (void*) &bufferrandZ[i]);
        clSetKernelArg(kernel_randGen[i], 1, sizeof(cl_mem), (void*) &bufferOutput[i]);
        clSetKernelArg(kernel_randGen[i], 2, sizeof(cl_mem), (void*) &bufferInput[i]);
    }

    // // Set input
    // result = clEnqueueWriteBuffer(command_queue, bufferInput, CL_FALSE, 0, sizeInput, input, 0, NULL, NULL);
    // if(result!=CL_SUCCESS) printOpenCLError("clEnqueueWriteBuffer", result);

    for (int i = 0; i < ndev; i++)
    {
        // Execute the kernel_randGen
        result = clEnqueueNDRangeKernel(command_queue[i], kernel_randGen[i], 1, NULL, global, local, 0, NULL, NULL);
        if(result!=CL_SUCCESS) printOpenCLError("clEnqueueNDRangeKernel", result);
    }

    for (int i = 0; i < ndev; i++)
    {
        // Wait for all commands in queue to finish
        result = clFinish(command_queue[i]);
        if(result != CL_SUCCESS) printOpenCLError("clFinish", result);
    }

    // // Copy output
    // result = clEnqueueReadBuffer(command_queue, bufferOutput, CL_TRUE, 0, sizeOutput, output, 0, NULL, NULL);
    // if(result != CL_SUCCESS) printOpenCLError("clEnqueueReadBuffer", result);
    // result = clEnqueueReadBuffer(command_queue, bufferrandZ, CL_TRUE, 0, sizerandZ, randZ, 0, NULL, NULL);
    // if(result != CL_SUCCESS) printOpenCLError("clEnqueueReadBuffer", result);
    
    // debug check output
    // for(int i=0; i<GLOBAL_WORK_ITEMS; i++)
    // {
    //    // if(output[i]) printf("%d: %lf\n", i, output[i]);
    //    printf("%d: %lf\n", i, randZ[i]);
    // }
#else
            for(int i=0; i<BLOCKSIZE*iN*iFactors; i++){
              randZ[i] = RanUnif(lRndSeed);
            }
#endif  //ENABLE_OPENCL
        //}
     // }
    //}

    // =====================================================
    // shocks to hit various factors for forward curve at t

#ifdef TBB_VERSION
    ParallelB B(pdZ, randZ, BLOCKSIZE, iN);
    for(l=0;l<=iFactors-1;++l){
      B.set_l(l);
      tbb::parallel_for(tbb::blocked_range<int>(0, BLOCKSIZE, PARALLEL_B_GRAINSIZE),B);
    }

#else
    #ifdef ENABLE_OPENCL

    for (int i = 0; i < ndev; i++)
    {
        // Set the arguments of the kernel_serialB
        clSetKernelArg(kernel_serialB[i], 0, sizeof(cl_mem), (void*) &bufferpdZ[i]);
        clSetKernelArg(kernel_serialB[i], 1, sizeof(cl_mem), (void*) &bufferrandZ[i]);
        clSetKernelArg(kernel_serialB[i], 2, sizeof(cl_mem), (void*) &bufferOutput[i]);
        clSetKernelArg(kernel_serialB[i], 3, sizeof(cl_mem), (void*) &bufferInput[i]);
    }
        
        // // Copy the input vectors to the corresponding buffers
        // result = clEnqueueWriteBuffer(command_queue, bufferInput, CL_TRUE, 0, sizeInput, input, 0, NULL, NULL);
        // if(result!=CL_SUCCESS) printOpenCLError("clEnqueueWriteBuffer", result);
        // result = clEnqueueWriteBuffer(command_queue, bufferrandZ, CL_TRUE, 0, sizerandZ, randZ, 0, NULL, NULL);
        // if(result!=CL_SUCCESS) printOpenCLError("clEnqueueWriteBuffer", result);
        
    for (int i = 0; i < ndev; i++)
    {
        // Execute the kernel_serialB
        result = clEnqueueNDRangeKernel(command_queue[i], kernel_serialB[i], 1, NULL, global, local, 0, NULL, NULL);
        if(result!=CL_SUCCESS) printOpenCLError("clEnqueueNDRangeKernel", result);
    }

    for (int i = 0; i < ndev; i++)
    {
        // Wait for all commands in queue to finish
        result = clFinish(command_queue[i]);
        if(result != CL_SUCCESS) printOpenCLError("clFinish", result);
    }

    for (int i = 0; i < ndev; i++)
    {
        // Copy the result from bufferOutput to output
        result = clEnqueueReadBuffer(command_queue[i], bufferOutput[i], CL_TRUE, 0, sizeOutput, output, 0, NULL, NULL);
        if(result != CL_SUCCESS) printOpenCLError("clEnqueueReadBuffer", result);
        // result = clEnqueueReadBuffer(command_queue, bufferpdZ, CL_TRUE, 0, sizepdZ, pdZ, 0, NULL, NULL);
        // if(result != CL_SUCCESS) printOpenCLError("clEnqueueReadBuffer", result);
    }

        //debug check output
        // for(int i=0; i<GLOBAL_WORK_ITEMS; i++)
        // {
        //    if(output[i]) printf("%d: %lf\n", i, output[i]);
        // }
        
    #else
    /* 18% of the total executition time */
    serialB(pdZ, randZ, BLOCKSIZE, iN, iFactors);

    #endif  //ENABLE_OPENCL

#endif

    // =====================================================
    // Generation of HJM Path1
#ifdef ENABLE_OPENCL
    for (int i = 0; i < ndev; i++)
    {
        clSetKernelArg(kernel_pathGen[i], 0, sizeof(cl_mem), (void*) &bufferppdHJMPath[i]);
        clSetKernelArg(kernel_pathGen[i], 1, sizeof(cl_mem), (void*) &bufferppdFactors[i]);
        clSetKernelArg(kernel_pathGen[i], 2, sizeof(cl_mem), (void*) &bufferpdZ[i]);
        clSetKernelArg(kernel_pathGen[i], 3, sizeof(cl_mem), (void*) &bufferpdTotalDrift[i]);
        clSetKernelArg(kernel_pathGen[i], 4, sizeof(cl_mem), (void*) &bufferOutput[i]);
        clSetKernelArg(kernel_pathGen[i], 5, sizeof(cl_mem), (void*) &bufferInput[i]);
    }

    for (int i = 0; i < ndev; i++)
    {
        // Set input
        // result = clEnqueueWriteBuffer(command_queue, bufferInput, CL_FALSE, 0, sizeInput, input, 0, NULL, NULL);
        // if(result!=CL_SUCCESS) printOpenCLError("clEnqueueWriteBuffer", result);
        // result = clEnqueueWriteBuffer(command_queue, bufferpdZ, CL_FALSE, 0, sizepdZ, pdZ, 0, NULL, NULL);
        // if(result!=CL_SUCCESS) printOpenCLError("clEnqueueWriteBuffer", result);
        result = clEnqueueWriteBuffer(command_queue[i], bufferppdFactors[i], CL_FALSE, 0, sizeppdFactors, ppdFactors, 0, NULL, NULL);
        if(result!=CL_SUCCESS) printOpenCLError("clEnqueueWriteBuffer", result);
        result = clEnqueueWriteBuffer(command_queue[i], bufferpdTotalDrift[i], CL_FALSE, 0, sizepdTotalDrift, pdTotalDrift, 0, NULL, NULL);
        if(result!=CL_SUCCESS) printOpenCLError("clEnqueueWriteBuffer", result);
    }

    for (int i = 0; i < ndev; i++)
    {
        // Execute the kernel_pathGen
        result = clEnqueueNDRangeKernel(command_queue[i], kernel_pathGen[i], 1, NULL, global, local, 0, NULL, NULL);
        if(result!=CL_SUCCESS) printOpenCLError("clEnqueueNDRangeKernel", result);
    }

    for (int i = 0; i < ndev; i++)
    {
        // Wait for all commands in queue to finish
        result = clFinish(command_queue[i]);
        if(result != CL_SUCCESS) printOpenCLError("clFinish", result);
    }

    for (int i = 0; i < ndev; i++)
    {
        // Copy the result from bufferOutput to output
        // result = clEnqueueReadBuffer(command_queue, bufferOutput, CL_TRUE, 0, sizeOutput, output, 0, NULL, NULL);
        // if(result != CL_SUCCESS) printOpenCLError("clEnqueueReadBuffer", result);
        result = clEnqueueReadBuffer(command_queue[i], bufferppdHJMPath[i], CL_TRUE, 0, sizeppdHJMPath, ppdHJMPath, 0, NULL, NULL);
        if(result != CL_SUCCESS) printOpenCLError("clEnqueueReadBuffer", result);
    }

    //debug check output
    // for(int i=0; i<GLOBAL_WORK_ITEMS; i++)
    // {
    //    if(output[i]) printf("%d: %lf\n", i, output[i]);
    // }

    for (int i = 0; i < ndev; i++)
    {
        // Wait for all commands in queue to finish
        result = clFinish(command_queue[i]);
        if(result != CL_SUCCESS) printOpenCLError("clFinish", result);
    }
#else
    for(int b=0; b<BLOCKSIZE; b++){ // b is the blocks
      for (j=1;j<=iN-1;++j) {// j is the timestep
        
        for (l=0;l<=iN-(j+1);++l){ // l is the future steps
          dTotalShock = 0;
          for (i=0;i<=iFactors-1;++i){// i steps through the stochastic factors
            dTotalShock += ppdFactors[(i*(iN-1))+l]* pdZ[(i*BLOCKSIZE*iN) + BLOCKSIZE*j + b];               
          }            

          ppdHJMPath[(j*BLOCKSIZE*iN)+BLOCKSIZE*l+b] = ppdHJMPath[((j-1)*BLOCKSIZE*iN)+BLOCKSIZE*(l+1)+b]+ pdTotalDrift[l]*ddelt + sqrt_ddelt*dTotalShock;
          //as per formula
        }
      }
    } // end Blocks
#endif  //ENABLE_OPENCL
    // -----------------------------------------------------

    free(ppdFactors);
    // free_dmatrix(pdZ, 0, iFactors -1, 0, iN*BLOCKSIZE -1);
    // free_dmatrix(randZ, 0, iFactors -1, 0, iN*BLOCKSIZE -1);
    iSuccess = 1;
    return iSuccess;
}
    


