/*
* cudnn.hpp
*
*  Created on: Dec 8, 2015
*      Author: tdx
*/

#ifndef CUDNN_HPP_
#define CUDNN_HPP_

#include<iostream>
#include<cuda_runtime.h>
#include<cudnn.h>
#include<cublas.h>
#include<cublas_v2.h>

#include"common/common.hpp"

using namespace std;

void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                   cudnnTensorFormat_t& tensorFormat, 
                   cudnnDataType_t& dataType,
                   int n,
                   int c,
                   int h,
                   int w);

void matrixMulti(cublasHandle_t cublasHandle, 
                 int m, 
                 int n, 
                 int batchSize, 
                 float alpha,
                 const float*A, 
                 const float*x, 
                 float beta, 
                 float *y);

template <typename Ntype>
class cuDNN
{
    public:
    cudnnDataType_t GetDataType() const { return  dataType; }

    cudnnTensorFormat_t GetTensorFormat() const { return tensorFormat; }

    cudnnHandle_t GetcudnnHandle() const { return cudnnHandle; }

    cublasHandle_t GetcublasHandle() const { return cublasHandle; }

    int getConvFwdAlgorithm() const { return convFwdAlgorithm; }

    int getConvolutionBwdFilterAlgorithm() const { return convBwdFilterAlgorithm; }

    int getConvolutionBwdDataAlgorithm() const { return convBwdDataAlgorithm; }

    void setConvolutionFwdAlgorithm(const cudnnConvolutionFwdAlgo_t& algo) { convFwdAlgorithm = static_cast<int>(algo); }

    void setConvolutionBwdFilterAlgorithm(const cudnnConvolutionBwdFilterAlgo_t& algo) { convBwdFilterAlgorithm = static_cast<int>(algo);}

    void setConvolutionBwdDataAlgorithm(const cudnnConvolutionBwdDataAlgo_t& algo) { convBwdDataAlgorithm = static_cast<int>(algo);}

    private:
    int convFwdAlgorithm;
    int convBwdFilterAlgorithm;
    int convBwdDataAlgorithm;
    // Inlcude 3 type：float32、double64、float16
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    void createHandles()
    {
        CUDNN_CHECK(cudnnCreate(&cudnnHandle));
        CUBLAS_CHECK(cublasCreate(&cublasHandle));
    }

    void destroyHandles()
    {
        CUDNN_CHECK(cudnnDestroy(cudnnHandle));
        CUBLAS_CHECK(cublasDestroy(cublasHandle));
    }

public:
    static cuDNN<Ntype>* getInstance()
    {
        static cuDNN<Ntype>* cudnn = new cuDNN<Ntype>();
        return cudnn;
    }
    cuDNN()
    {
    	convFwdAlgorithm = -1;
    	convBwdFilterAlgorithm = -1;
    	convBwdDataAlgorithm = -1;

        switch(sizeof(Ntype))
        {
            case 4:
            	dataType = CUDNN_DATA_FLOAT;break;
            case 8:
            	dataType = CUDNN_DATA_DOUBLE;break;
            case 2:
            	dataType = CUDNN_DATA_HALF;break;
            default:FatalError("Unsupported data type");break;
        }
        /*format type*/
        tensorFormat = CUDNN_TENSOR_NCHW;
        createHandles();
    }
    ~cuDNN()
    {
        destroyHandles();
    }
};

#endif /* CUDNN_NETWORK_H_ */
