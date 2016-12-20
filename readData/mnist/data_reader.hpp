/*************************************************************************
	> File Name: data_reader.hpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月19日 星期一 10时38分20秒
 ************************************************************************/

#ifndef _DATA_READER_HPP_
#define _DATA_READER_HPP_

#include<string>
#include"common/nDMatrix.hpp"

using namespace std;

uint32_t swap_endian(uint32_t val);
void readMnistData(NDMatrix<float>& dataX, NDMatrix<int>& labelY, string image_filename, string label_filename);

#endif
