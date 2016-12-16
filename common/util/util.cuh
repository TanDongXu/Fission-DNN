/*************************************************************************
	> File Name: util.cuh
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月14日 星期三 09时04分23秒
 ************************************************************************/

#ifndef _UTIL_CUH_
#define _UTIL_CUH_

#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>

#define ACTIVATION_SIGMOID 0
#define ACTIVATION_RELU 1
#define ACTIVATION_TANH 2;
#define AVTIVATION_CLIPPED_RELU 3
#define ACTIVATION_LRELU 4

#define POOL_MAX 0
#define POOL_AVERAGE_COUNT_INCLUDE_PADDING 1
#define POOL_AVERAGE_COUNT_EXCLUDE_PADDING 2

#define RAMDOM 1;
#define READ_FROM_FILE 2

#endif
