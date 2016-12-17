/*************************************************************************
	> File Name: layer.cpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月17日 星期六 09时54分26秒
 ************************************************************************/

#include<iostream>
#include"layer.hpp"
#include<glog/logging.h>
#include"config/configBase.hpp"

using namespace std;

template<typename Ntype>
inline Ntype Layer<Ntype>::Forward(const NDMatrix<Ntype>* bottom, const NDMatrix<Ntype>* top)
{
    Ntype loss = 0;
    Reshape(m_bottom);
    const string mode = ConfigTable::getInstance()->getSolver_mode();
    if(string("CPU") == mode)
    {
        Forward_cpu(bottom, top);
        // compute loss




    }else if(string("GPU") == mode)
    {
        Forward_gpu(bottom, top);
        // compute loss

        

    }else
        LOG(FATAL) << "Unknown Solver mode.";

    return loss;
}

template<typename Ntype>
inline void Layer<Ntype>::Backward(const NDMatrix<Ntype>* top, const NDMatrix<Ntype>* bottom)
{
    const string mode = ConfigTable::getInstance()->getSolver_mode();
    if(string("CPU") == mode)
    {
        Backward_cpu(top, bottom);
    }else if(string("GPU") == mode)
    {
        Backward_gpu(top, bottom);
    }else
        LOG(FATAL) << "Unknown Solver mode.";
}

