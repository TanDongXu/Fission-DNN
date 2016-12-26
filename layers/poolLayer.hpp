/*************************************************************************
	> File Name: poolLayer.hpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月23日 星期五 17时08分30秒
 ************************************************************************/

#ifndef _POOLLAYER_HPP_
#define _POOLLAYER_HPP_

#include<tuple>
#include<string>
#include<cudnn.h>
#include<math.h>
#include"layers/layer.hpp"

using namespace std;

/*
 * Class pool layer
 */
template<typename Ntype>
class PoolLayer : public Layer<Ntype>
{
    public:
    PoolLayer(string name);
    ~PoolLayer();
    Ntype Forward(Phase phase);
    void Backward();

    private:
    void ReShape();
    void createHandles();
    void destroyHandles();

    cudnnPoolingMode_t poolingMode;
    cudnnTensorDescriptor_t bottom_tensorDesc;
    cudnnTensorDescriptor_t top_tensorDesc;
    cudnnPoolingDescriptor_t poolingDesc;
    string m_pool_Type;
    int m_poolDim;
    int m_pad_h;
    int m_pad_w;
    int m_stride_h;
    int m_stride_w;
    int m_prev_num;
    int m_prev_channels;
    int m_prev_height;
    int m_prev_width;
};

#endif
