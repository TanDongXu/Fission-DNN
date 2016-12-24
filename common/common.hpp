/*************************************************************************
	> File Name: common.hpp
	> Author: TDX 
	> Mail: SA614149@mail.ustc.edu.cn
	> Created Time: 2016年12月10日 星期六 22时54分21秒
 ************************************************************************/

#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include <cudnn.h>
#include<cuda_runtime.h>
#include<glog/logging.h>
#include<stdlib.h>
#include<iostream>
#include<sstream>

// Instantiate a class width float and double specification
#define INSTANTIATE_CLASS(classname) \
    template class classname<float>; \
    template class classname<double>

#define CUDA_CHECK(condition) \
    do{ \
        cudaError_t error = condition; \
        CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
      }while(0)

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " "\
      << cudnnGetErrorString(status); \
  } while (0)

inline const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
  }
  return "Unknown cudnn status";
}

#define FatalError(s){                                                                      \
                      std::stringstream _where, _message;                                   \
                      _where << __FILE__<<':'<<__LINE__;                                    \
                      _message << std::string(s) + "\n" <<__FILE__ <<':'<<__LINE__;         \
                      std::cerr << _message.str() <<"\nAboring..\n";                        \
                      cudaDeviceReset();                                                    \
                      exit(EXIT_FAILURE);                                                   \
                     }

#define CUBLAS_CHECK(status){                                                                           \
                            std::stringstream _error;                                                   \
                            if(0 != status)                                                             \
                            {                                                                           \
                                _error<<"Cublas failure ERROR Code: "<<status;                           \
                                FatalError(_error.str());                                               \
                            }                                                                           \
                        }





#endif
