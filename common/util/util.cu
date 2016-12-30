#include<glog/logging.h>
#include<cmath>
#include<cstdlib>
#include<ctime>
#include<sstream>
#include"util.cuh"
#include"common/common.hpp"
#include"common/nDMatrix.hpp"

string int_to_string(int num)
{
    stringstream ss;
    ss << num;
    string s;
    s = ss.str();
    return s;
}

// ShowDevices information
void showDevices()
{
    int totalDevices;
    cudaGetDeviceCount(&totalDevices);
    LOG(INFO) << "There are " << totalDevices <<" CUDA Capable Devices on your machine: ";

    for(int i = 0; i < totalDevices; i++)
    {
        struct cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf( "device %d : sms %2d  Capabilities %d.%d, SmClock %.1f Mhz, MemSize (Mb) %d, MemClock %.1f Mhz, Ecc=%d, boardGroupID=%d\n",
               i,
               prop.multiProcessorCount,
               prop.major,
               prop.minor,
               (float)prop.clockRate*1e-3,
               (int)(prop.totalGlobalMem/(1024*1024)),
               (float)prop.memoryClockRate*1e-3,
               prop.ECCEnabled,
               prop.multiGpuBoardGroupID);
    }
    printf("\n");

}

/*
*Multi channels merge* 
*/
__global__ void MultiChannelsMerge(float** inputs,
                                   float* outputs, 
                                   int* channels, 
                                   int* indexs, 
                                   int row, 
                                   int outChannels)
{
    int batchId  = blockIdx.x;
    int index    = blockIdx.y;
    int offset   = indexs[index];
    int curChannels = channels[index];

    float *input  = inputs[index];
    float* output = outputs + batchId * outChannels * row * row + offset;

    int blockDo = curChannels * row * row;
    for(int i = 0; i < blockDo; i += blockDim.x)
    {
        int j = i + threadIdx.x;
        if (j < blockDo)
        {
            int pos = batchId * curChannels * row * row;
            output[j] = input[pos + j];
        }
    }
}

/*multiChannels data split*/
__global__ void MultiChannelsSplit(float* inputs, 
                                   float**outputs, 
                                   int* channels, 
                                   int* indexs, 
                                   int row, 
                                   int inChannels)
{
    int batchId  = blockIdx.x;
    int index    = blockIdx.y;
    int offset   = indexs[index];
    int curChannels = channels[index];

    float* output = outputs[index];
    float* input  = inputs + batchId * inChannels * row * row + offset;

    int blockDo = curChannels * row * row;
    for(int i = 0; i < blockDo; i += blockDim.x)
    {
        int j = i + threadIdx.x;
        if(j < blockDo)
        {
            int pos = batchId * curChannels * row * row;
            output[pos + j] = input[j];
        }
    }
}

/*overload*/
__global__ void MultiChannelsSplit(float* inputs, 
                                   float* outputs, 
                                   int outChannels, 
                                   int offset, 
                                   int row, 
                                   int inChannels)
{
    int  batchId = blockIdx.x;
    float* input = inputs + batchId * inChannels * row * row + offset;

    int blockDo  = outChannels * row * row;
    for(int i = 0; i < blockDo; i += blockDim.x)
    {
        int j = i + threadIdx.x;
        if(j < blockDo)
        {
            int pos = batchId * outChannels * row * row;
            outputs[pos + j] = input[j];
        }
    }
}

/*multi array add*/
__global__ void MultiArrayAdd(float** inputs, 
                              float* outputs, 
                              int number,
                              int channels, 
                              int height, 
                              int width)
{
    int blockDo = number * channels * height * width;
    for(int j = 0; j < 4; j++){
        float* input = inputs[j];
        for(int i = 0; i < blockDo; i += blockDim.x)
        {
            int idx = i + threadIdx.x;
            if(idx < blockDo)
            {
                outputs[idx] = outputs[idx] + input[idx];
            }
        }
    }
}

template<typename Ntype>
void createGaussian(NDMatrix<Ntype>* gaussian, float epsilon)
{
    int number = gaussian->ND_num();
    int channels = gaussian->ND_channels();
    int rows = gaussian->ND_height();
    int cols = gaussian->ND_width();

    float dElasticSigma1;
    float dElasticSigma2;

    int iiMidr = rows >> 1;
    int iiMidc = cols >> 1;
    Ntype* pGauss = gaussian->mutable_cpu_data();

    float _sum = 0.0;
    for(int num = 0; num < number; num++)
    {
        for(int ch = 0; ch < channels; ch++)
        {
            dElasticSigma1 = 0.5f + 4.0f * (rand()) / RAND_MAX;
            dElasticSigma2 = 0.5f + 4.0f * (rand()) / RAND_MAX;
            for(int row = 0; row < rows; row++)
            {
                for(int col = 0; col < cols; col++)
                {
                    float val1 = 1.0f / (dElasticSigma1 * dElasticSigma2 * 2.0f * 3.1415926535897932384626433832795f);
                    float val2 = 1.0f * (row-iiMidr)*(row-iiMidr) / (dElasticSigma1 * dElasticSigma1) + 1.0f * (col-iiMidc)*(col-iiMidc) / (dElasticSigma2 * dElasticSigma2) 
                        + 2.0f * (row - iiMidr) * (col - iiMidc) / (dElasticSigma1 * dElasticSigma2);
                    pGauss[gaussian->ND_offset(num, ch, row, col)] = val1 * exp(-1.0f * val2);
                    _sum += pGauss[gaussian->ND_offset(num, ch, row, col)];
                }
            }
        }
    }
        
    for(int num = 0; num < number; num++)
    {
        for(int ch = 0; ch < channels; ch++)
        {
            for(int row = 0; row < rows; row++)
            {
                for(int col = 0; col < cols; col++)
                {
                    float val = pGauss[gaussian->ND_offset(num, ch, row, col)] / _sum;
                    pGauss[gaussian->ND_offset(num, ch, row, col)] = val * epsilon;
                }       
            }   
        }
    }
}

template void createGaussian<float>(NDMatrix<float>* gaussian, float epsilon);
template void createGaussian<double>(NDMatrix<double>* gaussian, float epsilon);

