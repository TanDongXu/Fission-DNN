/*************************************************************************
	> File Name: test.cpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月19日 星期一 22时55分04秒
 ************************************************************************/
#include"test.hpp"
#include<iostream>
#include"common/nDMatrix.hpp"

using namespace std;

template<>
void printf_NDMatrix_data(NDMatrix<float>*matrix)
{
    const int num = matrix->ND_num();
    const int channels = matrix->ND_channels();
    const int height = matrix->ND_height();
    const int width = matrix->ND_width();

    for(int n = 0; n < num; n++)
    {
        for(int c = 0; c < channels; c++)
        {
            for(int h = 0; h < height; h++)
            {
                for(int  w = 0; w < width; w++)
                {
                    cout<< matrix->data_at(n, c, h, w)<<" ";
                }

                cout<<endl;
            }
        }

        cout<< endl;
        for(;;);
    }
}

template<>
void printf_NDMatrix_data(NDMatrix<int>*matrix)
{
    const int num = matrix->ND_num();
    const int channels = matrix->ND_channels();
    const int height = matrix->ND_height();
    const int width = matrix->ND_width();
    for(int n = 0; n < num; n++)
    {
        for(int c = 0; c < channels; c++)
        {
            for(int h = 0; h < height; h++)
            {
                for(int  w = 0; w < width; w++)
                {
                    cout<< matrix->data_at(n, c, h, w)<<" ";
                }

                cout<<endl;
            }
        }

        cout<< endl;
    for(;;);
    }
}
