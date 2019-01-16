#ifndef CUDA_HELPER_H
#define	CUDA_HELPER_H


//#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <cublas_v2.h>


/******************************************************************************
****** Constant parameters ****************************************************
*******************************************************************************/
#define NTHREADS 640

#define SPARSITY_PARAM 0.1f
#define BETA 3.0f
#define LAMBDA 0.003f // Weight decay parameter
#define EPS_COST 0.000001f // Training stops if |new_cost - old_cost| < EPS_COST

/******************************************************************************
****** Utilities **************************************************************
*******************************************************************************/
const char* cublasGetErrorString(cublasStatus_t status);

#define CUDA_SAFE_CALL(call) do { cudaError_t err = call; \
	if(err != cudaSuccess) { \
		printf("Error at %s:%d\nError: %s\n",__FILE__,__LINE__, cudaGetErrorString(err)); \
		exit(EXIT_FAILURE);}} while(0)

#define CUBLAS_SAFE_CALL(call) do { cublasStatus_t err = call; \
	if(err != CUBLAS_STATUS_SUCCESS) { \
		printf("Cublas error at %s:%d\nError: %s\n",__FILE__,__LINE__, cublasGetErrorString(call)); \
		exit(EXIT_FAILURE);}} while(0)

#define ERROR_REPORT(description) do { \
	printf("Error at %s:%d\nDescription: %s\n",__FILE__,__LINE__, description); \
    exit(EXIT_FAILURE);} while(0)

#define PROFILE(text, call) do {  \
	long startTime = clock(); \
	call; \
	long finishTime = clock(); \
	std::cout<< text << ": " << (finishTime - startTime) / (double) CLOCKS_PER_SEC << " seconds" << std::endl; \
	} while(0)

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

/******************************************************************************
****** My kernels <3 **********************************************************
*******************************************************************************/
// Initialize curand random number generator

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

//#endif
#endif	/* SPARSE_AUTOENCODER_HELPER_CUH */