/*************************************************************************
	> File Name: common.hpp
	> Author: TDX 
	> Mail: SA614149@mail.ustc.edu.cn
	> Created Time: 2016年12月10日 星期六 22时54分21秒
 ************************************************************************/

#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include<glog/logging.h>

// Instantiate a class width float and double specification
#define INSTANTIATE_CLASS(classname) \
    template class classname<float>; \
    template class classname<double>

#define CUDA_CHECK(condition) \
    do{ \
        cudaError_t error = condition; \
        CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
      }while(0)










#endif
