/*************************************************************************
	> File Name: dataLayer.hpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月19日 星期一 14时56分13秒
 ************************************************************************/

#ifndef _DATALAYER_HPP_
#define _DATALAYER_HPP_

#include<iostream>
#include<string>

#include"layers/layer.hpp"
#include"common/nDMatrix.hpp"
#include"dataTransFormer/data_transformer.hpp"
;
using namespace std;

template<typename Ntype>
class DataLayer:public Layer<Ntype>
{
    public:
    DataLayer(string name, const int rows, const int cols);
    ~DataLayer();
    void load_batch(int index, NDMatrix<Ntype>& image_data, NDMatrix<int>& image_labels );
    void random_load_batch(NDMatrix<Ntype>& image_data, NDMatrix<int>& image_labels);
    inline int getDataSize() const { return m_batchSize; }
    inline NDMatrix<int>* getLables() const { return m_labels; }
    Ntype Forward(Phase phase);
    void Backward();

    private:
    void ReShape();
    NDMatrix<int>* m_labels;
    DataTransformer<Ntype>* m_data_transformer;
    int m_batchSize;
    int m_dataSize;
    // preprocessing
    bool m_isDataTransFormer;
    bool m_doMirror;
    int m_cropSize;
    float m_scale;
};

#endif
