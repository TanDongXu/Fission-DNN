/*
* LRNLayer.hpp
*
*  Created on: Dec 31, 2015
*      Author: tdx
*/

#ifndef LRNLAYER_HPP_
#define LRNLAYER_HPP_

#include<string>
#include<cudnn.h>
#include"layer.hpp"

using namespace std;

/*
 * Class LRN layer
 */
template<typename Ntype>
class LRNLayer : public Layer<Ntype>
{
    public:
    LRNLayer(string name);
    ~LRNLayer();
    void Forward(Phase phase);
    void Backward();

    private:
    void ReShape();
    void createHandles();
    void destroyHandles();

    private:
    cudnnLRNDescriptor_t normDesc;
    cudnnTensorDescriptor_t bottom_tensorDesc;
    cudnnTensorDescriptor_t top_tensorDesc;
    unsigned lrnN ;
    double lrnAlpha;
    double lrnBeta;
    double lrnK;
};

#endif /* LRNLAYER_H_ */
