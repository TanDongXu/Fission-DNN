/*************************************************************************
	> File Name: layer.hpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月16日 星期五 14时45分01秒
 ************************************************************************/

#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include<iostream>
#include<vector>
#include<string>
#include"common/nDMatrix.hpp"
#include"config/configBase.hpp"

using namespace std;

enum Phase{ TRAIN, TEST };

template<typename Ntype>
class Layer
{
    public:
    virtual ~Layer(){}
    void setUp(const NDMatrix<Ntype>* bottom, const NDMatrix<Ntype>* top)
    {
        layerSetUp(m_bottom, m_top);
        ReShape(m_bottom, m_top);
    }
    virtual void layerSetUp(const NDMatrix<Ntype>* bottom, const NDMatrix<Ntype>* top) = 0;
    // Adjust the shapes of top NDMatrix and internal buffers to accommodate the shapes of the bottom
    virtual void ReShape(const NDMatrix<Ntype>* bottom, const NDMatrix<Ntype>* top) = 0;
    // Given the bottom NDMatrix, compute the top NDMatrix and the loss
    inline Ntype Forward(const NDMatrix<Ntype>* bottom, const NDMatrix<Ntype>* top);
    // given the top NDMatrix error gradients, compute the bottom NDMatrix error gradients
    inline void Backward(const NDMatrix<Ntype>* top, const NDMatrix<Ntype>* bottom);
    // Rerturn the layer type
    virtual inline const string type() const{ return""; }
    
    protected:
    // use cpu device compute the output
    virtual void Forward_cpu(const NDMatrix<Ntype>* bottom, const NDMatrix<Ntype>* top) = 0;
    // Use Gpu device compute the output
    virtual void Forward_gpu(const NDMatrix<Ntype>* bottom, const NDMatrix<Ntype>* top) = 0;
    // Use cpu device compute the gradients
    virtual void Backward_cpu(const NDMatrix<Ntype>* top, const NDMatrix<Ntype>* bottom) = 0;
    // Use Gpu device compute the gradients
    virtual void Backward_gpu(const NDMatrix<Ntype>* top, const NDMatrix<Ntype>* bottom) = 0;

    NDMatrix<Ntype>* m_bottom;
    NDMatrix<Ntype>* m_top;
    Ntype m_loss;
    vector<BaseLayerConfig*> m_prevLayer;
    vector<BaseLayerConfig*> m_nextLayer;
};

#endif
