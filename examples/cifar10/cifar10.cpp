#include<iostream>
#include<glog/logging.h>
#include<cudnn.h>

#include"common/nDMatrix.hpp"
#include"config/configBase.hpp"
#include"readData/cifar10/data_reader.hpp"
#include"common/util/util.cuh"
#include"net/net.hpp"

#include"cifar10.hpp"


void runCifar10()
{
	NDMatrix<float>  trainSetX;
	NDMatrix<float>  testSetX;
	NDMatrix<int> trainSetY, testSetY;
	// Read the layers config
	ConfigTable::getInstance()->initConfig("profile/cifar10/Cifar10Config.txt");
	// Read the cifar10 data
	readCifar10Data(trainSetX, testSetX, trainSetY, testSetY);

	LOG(INFO) << "*******************************************************";;
	LOG(INFO) << "     Train_set : "<< trainSetX.ND_channels() <<" x "<<trainSetX.ND_height() <<" x " <<trainSetX.ND_width() <<" features and "<< trainSetX.ND_num() <<" samples";
	LOG(INFO) << "   Train_label : "<< trainSetY.ND_channels() <<" x "<<trainSetY.ND_height() <<" x " <<trainSetY.ND_width() <<" features and "<< trainSetY.ND_num() <<" samples";
	LOG(INFO) << "      Test_set : "<< testSetX.ND_channels() <<" x "<<testSetX.ND_height() <<" x " <<testSetX.ND_width() <<" features and "<< testSetX.ND_num() <<" samples";
	LOG(INFO) << "    Test_label : "<< testSetY.ND_channels() <<" x "<<testSetY.ND_height() <<" x " <<testSetY.ND_width() <<" features and "<< testSetY.ND_num() <<" samples";
	LOG(INFO) << "*******************************************************";

	 int cuda_version = cudnnGetVersion();
	 LOG(INFO) << "cudnnGetVersion(): " << cuda_version << " CUDNN VERSION from cudnn.h: "<< CUDNN_VERSION;
	 /*show the device information*/
	 showDevices();

	// Create network 
    createNet(trainSetX.ND_height(), trainSetX.ND_width());
    // Training Network
	trainNetWork(trainSetX, trainSetY, testSetX, testSetY);

}
