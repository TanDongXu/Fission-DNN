#include<iostream>
#include<glog/logging.h>
#include<cudnn.h>

#include"runMnist.hpp"
#include"common/nDMatrix.hpp"
#include"config/configBase.hpp"
#include"readData/mnist/data_reader.hpp"
#include"common/util/util.cuh"
#include"net/net.hpp"

using namespace std;

void runMnist()
{
    NDMatrix<float> trainSetX;
    NDMatrix<float> testSetX;
    NDMatrix<int> trainSetY, testSetY;
	// Read the layers configure
    ConfigTable::getInstance()->initConfig("profile/mnist/MnistConfig.txt");

	// Read Mnist dataSet
	readMnistData(trainSetX, trainSetY, "data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte");
    readMnistData(testSetX, testSetY, "data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte");
    LOG(INFO) << "*******************************************************";
    LOG(INFO) << "     Train_set : " << trainSetX.ND_height() <<" x "<< trainSetX.ND_width() << " features and " << trainSetX.ND_num() << " samples";
    LOG(INFO) << "   Train_label :   " << trainSetY.ND_height() <<" x "<< trainSetY.ND_width() << " features and " << trainSetY.ND_num() << " samples";
    LOG(INFO) << "      Test_set : " << testSetX.ND_height() <<" x "<< testSetX.ND_width() << " features and " <<  testSetX.ND_num() << " samples";
    LOG(INFO) << "    Test_label :   " << testSetY.ND_height() <<" x "<< testSetY.ND_width() << " features and " <<  testSetY.ND_num()  << " samples";
    LOG(INFO) <<"*******************************************************";

    int cuda_version = cudnnGetVersion();
    LOG(INFO) << "cudnnGetVersion(): " << cuda_version << " CUDNN VERSION from cudnn.h: " << CUDNN_VERSION;
    // Show the device information
    showDevices();
    // Create network
    createNet(trainSetX.ND_height(), trainSetX.ND_width());
    // Training Network
    trainNetWork(trainSetX, trainSetY, testSetX, testSetY);
}
