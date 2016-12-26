/*************************************************************************
	> File Name: convLayer.hpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月22日 星期四 17时31分02秒
 ************************************************************************/

#ifndef _CONVLAYER_HPP_
#define _CONVLAYER_HPP_

#include<cudnn.h>
#include<curand.h>
#include"layer.hpp"
#include"common/nDMatrix.hpp"


template<typename Ntype>
class ConvLayer : public Layer<Ntype>
{
    public:
    ConvLayer(string name);
    ~ConvLayer();
    Ntype Forward(Phase phase);
    void Backward();

    private:
    void ReShape();
    void initRandom(bool isGaussian);
    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, int c, Ntype* data);
    void createHandles();
    void destroyHandles();

    NDMatrix<Ntype>* m_weight;
    NDMatrix<Ntype>* m_bias;
    float* tmp_Wgrad, *tmp_Bgrad;
    float m_lambda;
    float m_epsilon;
    float m_momentum;
    int m_batchSize;
    int m_kernelSize;
    int m_pad_h;
    int m_pad_w;
    int m_stride_h;
    int m_stride_w;
    int m_kernelAmount;
    int m_prev_num;
    int m_prev_channels;
    int m_prev_height;
    int m_prev_width;

    cudnnTensorDescriptor_t bottom_tensorDesc;
    cudnnTensorDescriptor_t top_tensorDesc;
    cudnnTensorDescriptor_t biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t convFwdAlgo;
    cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
    cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;
    curandGenerator_t curandGenerator_W;
    curandGenerator_t curandGenerator_B;
};

#endif
