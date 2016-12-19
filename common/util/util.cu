#include<glog/logging.h>
#include"util.cuh"
#include"common/common.hpp"

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
