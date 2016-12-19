/*************************************************************************
	> File Name: data_reader.cpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月18日 星期日 13时17分45秒
 ************************************************************************/

#include<iostream>
#include<fstream>
#include<string>
#include<glog/logging.h>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include"data_reader.hpp"


using namespace std;
using namespace cv;

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void readMnistData(vector<Mat>& dataX, Mat& labelY, string image_filename, string label_filename)
{
    ifstream image_file(image_filename, ios::in | ios::binary);
    ifstream label_file(label_filename, ios::in | ios::binary);
    CHECK(image_file) << "Unable to open file " << image_filename;
    CHECK(label_file) << "Unable to open file " << label_filename;

    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    CHECK_EQ(num_items,num_labels);
    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    labelY = Mat::zeros(1, num_labels, CV_8UC1);
    for(int item_id = 0; item_id < num_items; ++item_id)
    {
        Mat tmpMat = Mat::zeros(rows, cols, CV_8UC1);
        for(int r = 0; r < rows; ++r)
        {
            for(int c = 0; c < cols; ++c)
            {
                uchar temp = 0;
                image_file.read((char*)&temp, sizeof(temp));
                tmpMat.at<uchar>(r, c) = temp;
            }
        }
        dataX.push_back(tmpMat);

        //read labels
        uchar ltmp = 0;
        label_file.read((char*)&ltmp, sizeof(ltmp));
        labelY.at<uchar>(0, item_id) = (uchar)ltmp;
    }
}

