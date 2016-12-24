/*************************************************************************
	> File Name: net.hpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月20日 星期二 22时52分54秒
 ************************************************************************/

#ifndef _NET_HPP_
#define _NET_HPP_

#include"common/nDMatrix.hpp"

void createNet(const int rows, const int cols);
void trainNetWork(NDMatrix<float>& trainSetX, NDMatrix<int>& trainSetY, 
                  NDMatrix<float>& testSetX, NDMatrix<int>& testSetY);

#endif
