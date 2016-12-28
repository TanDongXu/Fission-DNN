/*
 * readCifar10Data.h
 *
 *  Created on: Mar 16, 2016
 *      Author: tdx
 */

#ifndef DATA_READER_HPP_
#define DATA_READER_HPP_


#include<string>
#include<sstream>
#include<fstream>
#include"common/nDMatrix.hpp"

void readCifar10Data(NDMatrix<float>& trainX,
		            NDMatrix<float>& testX,
		            NDMatrix<int>& trainY,
		            NDMatrix<int>& testY);

#endif /* READCIFAR10DATA_H_ */
