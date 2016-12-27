/*************************************************************************
	> File Name: test.hpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月19日 星期一 22时54分26秒
 ************************************************************************/

#ifndef _TEST_HPP_
#define _TEST_HPP_

#include<iostream>
#include"../common/nDMatrix.hpp"

template<typename Ntype>
void printf_NDMatrix_data(NDMatrix<Ntype>* matrix);

void printf_devData(int number, int channels, int height, int width, float* A);
template<typename Ntype>
void printf_NDMatrix_diff(NDMatrix<Ntype>* matrix);

#endif
