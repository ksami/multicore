#include <assert.h>
#include "HJM_type.h"

#include <cstring>

FTYPE RanUnif( long *s );
FTYPE CumNormalInv( FTYPE u );
void icdf_SSE(const int N, FTYPE *in, FTYPE *out);
void icdf_baseline(const int N, FTYPE *in, FTYPE *out);
int HJM_SimPath_Forward_SSE(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdForward, FTYPE *pdTotalDrift,
			    FTYPE **ppdFactors, long *lRndSeed);
int Discount_Factors_SSE(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath);
int Discount_Factors_opt(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath);


int HJM_SimPath_Forward_Blocking_SSE(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdForward, FTYPE *pdTotalDrift,
			    FTYPE **ppdFactors, long *lRndSeed, int BLOCKSIZE);
int HJM_SimPath_Forward_Blocking(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdForward, FTYPE *pdTotalDrift,
			    FTYPE **ppdFactors, long *lRndSeed, int BLOCKSIZE);


int Discount_Factors_Blocking(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath, int BLOCKSIZE);
int Discount_Factors_Blocking_SSE(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath, int BLOCKSIZE);


int HJM_Swaption_Blocking_SSE(FTYPE *pdSwaptionPrice, //Output vector that will store simulation results in the form:
			                              //Swaption Price
			                              //Swaption Standard Error
			      //Swaption Parameters 
			      FTYPE dStrike,				  
			      FTYPE dCompounding,     //Compounding convention used for quoting the strike (0 => continuous,
			      //0.5 => semi-annual, 1 => annual).
			      FTYPE dMaturity,	      //Maturity of the swaption (time to expiration)
			      FTYPE dTenor,	      //Tenor of the swap
			      FTYPE dPaymentInterval, //frequency of swap payments e.g. dPaymentInterval = 0.5 implies a swap payment every half
		                              //year
			      //HJM Framework Parameters (please refer HJM.cpp for explanation of variables and functions)
			      int iN,						
			      int iFactors, 
			      FTYPE dYears, 
			      FTYPE *pdYield, 
			      FTYPE **ppdFactors,
			      //Simulation Parameters
			      long iRndSeed, 
			      long lTrials, int blocksize, int tid);
 
int HJM_Swaption_Blocking(FTYPE *pdSwaptionPrice, //Output vector that will store simulation results in the form:
			                              //Swaption Price
			                              //Swaption Standard Error
			      //Swaption Parameters 
			      FTYPE dStrike,				  
			      FTYPE dCompounding,     //Compounding convention used for quoting the strike (0 => continuous,
			      //0.5 => semi-annual, 1 => annual).
			      FTYPE dMaturity,	      //Maturity of the swaption (time to expiration)
			      FTYPE dTenor,	      //Tenor of the swap
			      FTYPE dPaymentInterval, //frequency of swap payments e.g. dPaymentInterval = 0.5 implies a swap payment every half
		                              //year
			      //HJM Framework Parameters (please refer HJM.cpp for explanation of variables and functions)
			      int iN,						
			      int iFactors, 
			      FTYPE dYears, 
			      FTYPE *pdYield, 
			      FTYPE **ppdFactors,
			      //Simulation Parameters
			      long iRndSeed, 
			      long lTrials, int blocksize, int tid);
/*
extern "C" FTYPE *dvector( long nl, long nh );
extern "C" FTYPE **dmatrix( long nrl, long nrh, long ncl, long nch );
extern "C" void free_dvector( FTYPE *v, long nl, long nh );
extern "C" void free_dmatrix( FTYPE **m, long nrl, long nrh, long ncl, long nch );
*/

#ifdef ENABLE_OPENCL
const char* kernel_name = "test";
const char* program_src =
"__kernel void test(__global int* output) {"
"    int id = get_global_id(0);"
"    output[id]=id;"
"}";

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
// --OpenCL errors-- //
#endif  //ENABLE_OPENCL