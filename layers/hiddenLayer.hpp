/*
 * hiddenLayer.hpp
 *
 *  Created on: Nov 28, 2015
 *      Author: tdx
 */

#ifndef HIDDENLAYER_HPP_
#define HIDDENLAYER_H_

#include"layer.hpp"
#include"common/nDMatrix.hpp"
#include"config/configBase.hpp"
#include"common/cudnn.hpp"
#include"test/test.hpp"
#include<cuda_runtime.h>
#include<math.h>
#include "curand.h"

/*
 * Class Hidden layer
 */
template<typename Ntype>
class HiddenLayer: public Layer<Ntype>
{
public:
	HiddenLayer(string name);
	~HiddenLayer();
    void Forward(Phase phase);
    void Backward();

    private:
    void ReShape();
	void initRandom(bool isGaussian);
	void createHandles();
	void destroyHandles();

private:

    curandGenerator_t curandGenerator_W;
    curandGenerator_t curandGenerator_B;
    NDMatrix<Ntype>* m_weight;
    NDMatrix<Ntype>* m_bias;
    float* tmp_Wgrad, *tmp_Bgrad;
	float m_epsilon;
	float* VectorOnes;
	int m_inputSize;
    int m_outputSize;
	int m_batchSize;
	int m_prev_num;
	int m_prev_channels;
	int m_prev_height;
	int m_prev_width;
	float m_lambda;
    float m_momentum;
};



#endif /* HIDDENLAYER_H_ */
