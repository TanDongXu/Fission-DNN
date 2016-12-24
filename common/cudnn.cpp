#include"cudnn.hpp"
#include<cuda_runtime.h>
#include<cudnn.h>
#include<cublas.h>
#include<cublas_v2.h>
#include"common/common.hpp"



//#define ND_TENSOR_DESCRIPTOR

/*cudnn set tensor dim*/
void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                   cudnnTensorFormat_t& tensorFormat, 
                   cudnnDataType_t& dataType,
                   int n,
                   int c,
                   int h,
                   int w){

                       #if SIMPLE_TENSOR_DESCRIPTOR
                       /*cudnn set 4d tensor*/
                       CUDNN_CHECK(cudnnSetTensor4dDescriptor(tensorDesc, 
                                                             tensorFormat, 
                                                             dataType, 
                                                             n,
                                                             c,
                                                             h, 
                                                             w));

                       #elif defined(ND_TENSOR_DESCRIPTOR)

                       const int nDim = 4;
                       int dimA[nDim] = {n,c,h,w};
                       int strideA[nDim] = {c*h*w, h*w, w, 1};
                       CUDNN_CHECK(cudnnSetTensorNdDescriptor(tensorDesc,
                                                             dataType, 
                                                             4, 
                                                             dimA, 
                                                             strideA));

                       #else
                       CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(tensorDesc, 
                                                               dataType, 
                                                               n, 
                                                               c, 
                                                               h, 
                                                               w, 
                                                               c*h*w, 
                                                               h*w, 
                                                               w, 
                                                               1));

                       #endif
                   }

/*matrixMulti*/
#define DISABLE_GEMV
void matrixMulti(cublasHandle_t cublasHandle, 
                 int m, 
                 int n,
                 int batchSize, 
                 float alpha,
                 const float*A, 
                 const float*x, 
                 float beta, 
                 float *y)
{
    #ifdef DISABLE_GEMV
    CUBLAS_CHECK(cublasSgemm(cublasHandle,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_T,
                                  n,
                                  batchSize,
                                  m,
                                  &alpha,
                                  x,
                                  m,
                                  A,
                                  batchSize,
                                  &beta,
                                  y,
                                  n));

    #else
    CUBLAS_CHECK(cublasSgemv(cublasHandle, CUBLAS_OP_T,
                                  m, n,
                                  &alpha,
                                  A, m,
                                  x, 1,
                                  &beta,
                                  y, 1));
    #endif
}







