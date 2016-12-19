#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<glog/logging.h>
#include<cudnn.h>

#include"runMnist.hpp"
#include"config/configBase.hpp"
#include"readData/mnist/data_reader.hpp"
#include"common/util/util.cuh"

using namespace std;
using namespace cv;

void runMnist()
{
    vector<Mat> trainSetX;
    vector<Mat> testSetX;
    Mat trainSetY, testSetY;
	// Read the layers configure
    ConfigTable::getInstance()->initConfig("profile/mnist/MnistConfig.txt");

	// Read Mnist dataSet
	readMnistData(trainSetX, trainSetY, "data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte");
    readMnistData(testSetX, testSetY, "data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte");
    LOG(INFO) << "*******************************************************";
    LOG(INFO) << "     Train_set : " << trainSetX[0].rows <<" x "<< trainSetX[0].cols << " features and " << trainSetX.size() << " samples";
    LOG(INFO) << "   Train_label :   " << trainSetY.rows <<" x "<< trainSetY.rows << " features and " << trainSetY.cols  << " samples";
    LOG(INFO) << "      Test_set : " << testSetX[0].rows <<" x "<< testSetX[0].cols << " features and " <<  testSetX.size() << " samples";
    LOG(INFO) << "    Test_label :   " << testSetY.rows <<" x "<< testSetY.rows << " features and " <<  testSetY.cols  << " samples";
    LOG(INFO) <<"*******************************************************";

    int cuda_version = cudnnGetVersion();
    LOG(INFO) << "cudnnGetVersion(): " << cuda_version << " CUDNN VERSION from cudnn.h: " << CUDNN_VERSION;
    // Show the device information
    showDevices();

   // cout<<endl<<endl<<"Select the way to initial Parameter: "<<endl<<"1.random   2.read from file"<<endl;
   // int cmd;
   // cin>> cmd;
   // if(cmd == 1 || cmd == 2)
   // 	creatColumnNet(cmd);
   // else
   // {
   // 	cout<<"Init way input Error"<<endl;
   //     exit(0);
   // }

   // /*training Network*/
   // cuTrainNetWork(trainSetX, trainSetY, testSetX, testSetY, batchSize);
}
