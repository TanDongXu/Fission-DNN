/*
* main.cpp
*
*  Created on: Nov 19, 2015
*      Author: tdx
*/
#include<iostream>
#include<glog/logging.h>
#include"common/nDMatrix.hpp"
#include"config/configBase.hpp"
#include"examples/mnist/convert_mnist_data.hpp"
#include"examples/mnist/runMnist.hpp"
#include"examples/cifar10/cifar10.hpp"

int main(int argc, char** argv)
{
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging(argv[0]);

    cudaSetDevice(1);

    LOG(INFO) <<"Select the DataSet to Run:";
    LOG(INFO) <<"1.MNIST      2.CIFAR-10";
    int cmd;
    cin>>cmd;
    if(1 == cmd)
        runMnist();
    else if(2 == cmd)
        runCifar10();
    else
        LOG(FATAL) << "DataSet Select Error.";

    return 0;
}



