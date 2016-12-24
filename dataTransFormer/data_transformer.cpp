/*************************************************************************
	> File Name: data_transformer.cpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月17日 星期六 15时35分42秒
 ************************************************************************/

#include<iostream>
#include<glog/logging.h>
#include<stdlib.h>
#include<time.h>
#include<common/common.hpp>
#include"data_transformer.hpp"
#include"common/nDMatrix.hpp"
#include"common/math_function.hpp"

using namespace std;

template<typename Ntype>
void DataTransformer<Ntype>::Transform(NDMatrix<Ntype>*& input_NDMatrix, NDMatrix<Ntype>*& transformed_NDMatrix, Phase phase)
{
    const int input_num = input_NDMatrix->ND_num();
    const int input_channels = input_NDMatrix->ND_channels();
    const int input_height = input_NDMatrix->ND_height();
    const int input_width = input_NDMatrix->ND_width();

    const int num = transformed_NDMatrix->ND_num();
    const int channels = transformed_NDMatrix->ND_channels();
    const int height = transformed_NDMatrix->ND_height();
    const int width = transformed_NDMatrix->ND_width();
    const int size = transformed_NDMatrix->ND_count();

    CHECK_EQ(input_num, num);
    CHECK_EQ(input_channels, channels);
    CHECK_GE(input_height, height);
    CHECK_GE(input_width, width);

    int h_off = 0;
    int w_off = 0;
    if(m_cropSize)
    {
        CHECK_EQ(m_cropSize, height);
        CHECK_EQ(m_cropSize, width);
        //we only do random crop when we do training
        if(phase == TRAIN)
        {
            h_off = Rand(input_height - m_cropSize + 1);
            w_off = Rand(input_width - m_cropSize + 1);
        }else
        {
            h_off = (input_height - m_cropSize) / 2;
            w_off = (input_width - m_cropSize) / 2;
        }
    }else
    {
        CHECK_EQ(input_height, height);
        CHECK_EQ(input_width, width);
    }

    Ntype* input_data = input_NDMatrix->mutable_cpu_data();
    Ntype* transfromed_data = transformed_NDMatrix->mutable_cpu_data();
    for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (m_doMirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transfromed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transfromed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (m_scale != Ntype(1)) {
    cpu_scal<Ntype>(size, m_scale, transfromed_data);
  }

}

template<typename Ntype>
int DataTransformer<Ntype>::Rand(int n)
{
    srand((unsigned)time(NULL));
    return (rand() % n);
}


INSTANTIATE_CLASS(DataTransformer);
