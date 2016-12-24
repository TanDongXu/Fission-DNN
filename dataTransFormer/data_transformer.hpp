/*************************************************************************
	> File Name: data_transformer.hpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月17日 星期六 14时27分49秒
 ************************************************************************/

#ifndef _DATA_TRANSFORMER_HPP_
#define _DATA_TRANSFORMER_HPP_

#include<iostream>
#include<vector>
#include"layers/layer.hpp"

using namespace std;

// Applies common transformations to the input data, such assert
//  scaling, mirroring, substracting the image mean...
template<typename Ntype>
class DataTransformer
{
    public:
    explicit DataTransformer(int cropSize, bool do_mirror, Ntype scale):
                             m_cropSize(cropSize), m_doMirror(do_mirror), m_scale(scale){}
    ~DataTransformer(){}
    void Transform(NDMatrix<Ntype>*& input_NDMatrix, NDMatrix<Ntype>*& transformed_NDMatrix, Phase phase);

    private:
    // generate a random integer from Uniform({0,1,......n-1}).
    int Rand(int n);
    // Random number generator
    int m_cropSize;
    bool m_doMirror;
    Ntype m_scale;
};

#endif
