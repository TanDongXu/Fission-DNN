/*
* SoftMaxLayer.hpp
*
*  Created on: Nov 28, 2015
*      Author: tdx
*/

#ifndef SOFTMAXLAYER_HPP_
#define SOFTMAXLAYER_HPP_

#include<cudnn.h>
#include"layer.hpp"
#include"common/nDMatrix.hpp"
#include"dataLayer.hpp"
#include"config/configBase.hpp"
#include"common/cudnn.hpp"
#include"test/test.hpp"

/*
 *Class Softmax layer
 */
template<typename Ntype>
class SoftMaxLayer : public Layer<Ntype>
{
    public:
    SoftMaxLayer(string name);
    //SoftMaxLayer(const SoftMaxLayer* layer);
    ~SoftMaxLayer();
    Ntype Forward(Phase phase);
    void Backward();

    private:
    void ReShape();
    void initRandom();
    void ClassificationResults();
    void getBackPropDiffData();
    void getBatch_labels();
    void createHandles();
    void destroyHandles();

    cudnnTensorDescriptor_t bottom_tensorDesc;
    cudnnTensorDescriptor_t top_tensorDesc;
    int m_nClasses;
    int m_batchSize;
    int m_dataSize;
    NDMatrix<int>* m_srcLabels;
    int m_curCorrectSize;
    int m_correctSize;
    int m_flag;
};
#endif /* SOFTMAXLAYER_H_ */
