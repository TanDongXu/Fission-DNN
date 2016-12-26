/*
* DropOutLayer.h
*
*  Created on: Mar 15, 2016
*      Author: tdx
*/

#ifndef DROPOUTLAYER_H_
#define DROPOUTLAYER_H_

#include"layer.hpp"
#include<curand.h>

/*
 * Class DropOut layer
 */
template<typename Ntype>
class DropOutLayer : public Layer<Ntype>
{
    public:
    DropOutLayer(string name);
    ~DropOutLayer();
    Ntype Forward(Phase phase);
    void Backward();

    private:
    void ReShape();
    void CreateUniform(int size);
    void Dropout_TrainSet(float* data, int size, float dropout_rate);
    void Dropout_TestSet(float* data, int size, float dropout_rate);
    void createHandles();
    void destroyHandles();

    float DropOut_rate;
    float* outputPtr;
    curandGenerator_t curandGenerator_DropOut;
};


#endif /* DROPOUTLAYER_H_ */
