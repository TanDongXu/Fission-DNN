/*
* activationLayer.hpp
*
*  Created on: Dec 13, 2015
*      Author: tdx
*/

#ifndef ACTIVATIONLAYER_HPP_
#define ACTIVATIONLAYER_HPP_

#include"layer.hpp"
#include<cudnn.h>

/*
 * Class activation layer
 */
template<typename Ntype>
class ActivationLayer : public Layer<Ntype>
{
    public:
    ActivationLayer(string name);
    ~ActivationLayer();
    void Forward(Phase Phase);
    void Backward();

    private:
    void ReShape();
    void createHandles();
    void destroyHandles();

    int ActivationMode;
    cudnnActivationMode_t cudnnActivationMode;
    cudnnTensorDescriptor_t bottom_tensorDesc;
    cudnnTensorDescriptor_t top_tensorDesc;
    cudnnActivationDescriptor_t activDesc;
};

#endif /* ACTIVATIONLAYER_H_ */
