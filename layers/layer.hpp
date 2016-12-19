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
    // Adjust the shapes of top NDMatrix and internal buffers to accommodate the shapes of the bottom
    virtual void ReShape() = 0;
    // Given the bottom NDMatrix, compute the top NDMatrix and the loss
    virtual Ntype Forward() = 0;
    // given the top NDMatrix error gradients, compute the bottom NDMatrix error gradients
    virtual void Backward() = 0;
    // Rerturn the layer type
    virtual inline const string type() const{ return""; }
    
    protected:
    NDMatrix<Ntype>* m_bottom;
    NDMatrix<Ntype>* m_top;
    Ntype m_loss;
    vector<BaseLayerConfig*> m_prevLayer;
    vector<BaseLayerConfig*> m_nextLayer;
};

#endif
