#include"data_reader.hpp"
#include"common/util/util.cuh"
#include"glog/logging.h"

void read_batch(int index, string fileName, NDMatrix<float>& image_data, NDMatrix<int>& image_label)
{
    ifstream file(fileName, ios::binary);

    if(file.is_open())
    {
        int number_of_images = 10000;
        int n_channels = 3;
        int n_rows = 32;
        int n_cols = 32;

        float* pData = image_data.mutable_cpu_data();
        int* pLabels = image_label.mutable_cpu_data();

        int startIndex = index * number_of_images;
        for(int i = 0; i < number_of_images; i++)
        {
            unsigned char label;
            file.read((char*)&label, sizeof(label));
            pLabels[startIndex + i] = (int) label;
            
            for(int ch = 0; ch < n_channels; ch++)
            {
                for(int r = 0; r < n_rows; r++)
                {
                    for(int c = 0; c < n_cols; c++)
                    {
                        unsigned char temp = 0;
                        file.read((char*)&temp, sizeof(temp));
                        pData[c + n_cols * r + n_cols * n_rows * (i + startIndex)] = (float)(2.0f * temp / 255.0f - 1.0f);
                    }
                }
            }
        }
    }else
    {
        LOG(FATAL) << "Can not open file " << fileName;
    }
}

void readCifar10Data(NDMatrix<float>& trainX,
                    NDMatrix<float>& testX,
                    NDMatrix<int>& trainY,
                    NDMatrix<int>& testY)
{
    // readf the train data and label
    string file_dir = "data/cifar10/data_batch_";
    string suffix = ".bin";

    trainX.ND_reShape(50000, 3, 32, 32);
    trainY.ND_reShape(50000, 1, 1, 1);
    trainX.diff().reset();
    trainY.diff().reset();
    for(int i = 0; i < 5; i++)
    {
        string fileName = file_dir + int_to_string(i+1) + suffix;
        read_batch(i, fileName, trainX, trainY);
    }

    // read the test data and label
    file_dir = "data/cifar10/test_batch.bin";
    testX.ND_reShape(10000,3, 32, 32);
    testY.ND_reShape(10000, 1, 1, 1);
    testX.diff().reset();
    testY.diff().reset();
    read_batch(0, file_dir, testX, testY);
}
