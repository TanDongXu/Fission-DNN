#include"dropOutLayer.hpp"
#include"common/syncedmem.hpp"

template<typename Ntype>
void DropOutLayer<Ntype>::ReShape()
{
    this->m_top = this->m_bottom;
}

/*
 * DropOut layer constructor
 */
template<typename Ntype>
DropOutLayer<Ntype>::DropOutLayer(string name)
{
    this->m_name = name;
    this->m_inputName = " ";
    this->m_bottom = NULL;
    this->m_top = NULL;
    this->m_nextLayer.clear();
    this->m_prevLayer.clear();
    outputPtr = NULL;

    DropOutLayerConfig* curConfig = (DropOutLayerConfig*) ConfigTable::getInstance()->getLayerByName(this->m_name);
    string prevLayerName = curConfig->getInput();
    Layer<Ntype>* prev_Layer = (Layer<Ntype>*) LayerContainer<Ntype>::getInstance()->getLayerByName(prevLayerName);

    this->m_bottom = prev_Layer->getTop();
    CHECK(this->m_bottom);
    this->m_inputChannels = this->m_bottom->ND_channels();
    this->m_number = this->m_bottom->ND_num();
    this->m_channels = this->m_bottom->ND_channels();
    this->m_height = this->m_bottom->ND_height();
    this->m_width = this->m_bottom->ND_width();
    DropOut_rate = curConfig->getDropOut_rate();

    mallocDeviceMem((void**) &outputPtr, this->m_number * this->m_channels * this->m_height * this->m_width * sizeof(float));
    ReShape();
    this->createHandles();
}

/*
 * overload constructor
 * */
//DropOutLayer::DropOutLayer(const DropOutLayer* layer)
//{
//    srcData = NULL;
//    dstData = NULL;
//    nextLayer.clear();
//    prevLayer.clear();
//    outputPtr = NULL;
//
//    static int idx = 0;
//    _name = layer->_name + string("_") + int_to_string(idx);
//    idx ++;
//    _inputName = layer->_inputName;
//
//    inputAmount = layer->inputAmount;
//    inputImageDim = layer->inputImageDim;
//    number = layer->number;
//    channels = layer->channels;
//    height = layer->height;
//    width = layer->width;
//    outputSize = layer->outputSize;
//    DropOut_rate = layer->DropOut_rate;
//
//    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &outputPtr, number * channels * height * width * sizeof(float));
//
//    this->createHandles();
//    cout<<"Drop-copy"<<endl;
//}

/*
 * Destructor
 */
template<typename Ntype>
DropOutLayer<Ntype>::~DropOutLayer()
{
    freeDeviceMem(outputPtr);
    destroyHandles();
}

/*
 * Create Random handle
 */
template<typename Ntype>
void DropOutLayer<Ntype>::createHandles()
{
    curandCreateGenerator(&curandGenerator_DropOut, CURAND_RNG_PSEUDO_MTGP32);
}

__global__ void dropout_train(float* data, float* outputPtr, int size, float probability)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size)
    {
        if(outputPtr[idx] < probability)
        data[idx] = 0;
    }
}

__global__ void dropout_test(float* data, int size, float probability)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size)
    {
        data[idx] = data[idx] * probability;
    }
}

template<typename Ntype>
void DropOutLayer<Ntype>::CreateUniform(int size)
{
    curandSetPseudoRandomGeneratorSeed(curandGenerator_DropOut, time(NULL));
    curandGenerateUniform(curandGenerator_DropOut, outputPtr, size);
}

template<typename Ntype>
void DropOutLayer<Ntype>::Dropout_TrainSet(float* data, int size, float dropout_rate)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    dropout_train<<<blocksPerGrid, threadsPerBlock>>>(data, outputPtr, size, dropout_rate);
    cudaThreadSynchronize();
}

template<typename Ntype>
void DropOutLayer<Ntype>::Dropout_TestSet(float* data, int size, float dropout_rate)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    dropout_test<<<blocksPerGrid, threadsPerBlock>>>(data, size, DropOut_rate);
    cudaThreadSynchronize();
}

template<typename Ntype>
Ntype DropOutLayer<Ntype>::Forward(Phase phase)
{
    this->m_bottom = this->m_prevLayer[0]->getTop();

    /*use dropout in training, when testing multiply probability*/
    if(TRAIN == phase)
    {
        CreateUniform(this->m_number * this->m_channels * this->m_height * this->m_width);
        Dropout_TrainSet((float*)this->m_top->mutable_gpu_data(), this->m_number * this->m_channels * this->m_height * this->m_width, DropOut_rate);
    }
    else
    Dropout_TestSet((float*)this->m_top->mutable_gpu_data(), this->m_number * this->m_channels * this->m_height * this->m_width, DropOut_rate);
    
    return this->m_loss;
}

template<typename Ntype>
void DropOutLayer<Ntype>::Backward()
{
//    int nIndex = m_nCurBranchIndex;
//    diffData = nextLayer[nIndex]->diffData;
    Dropout_TrainSet((float*)this->m_bottom->mutable_gpu_diff(), this->m_number * this->m_channels * this->m_height * this->m_width, DropOut_rate);
}

template<typename Ntype>
void DropOutLayer<Ntype>::destroyHandles()
{
    curandDestroyGenerator(curandGenerator_DropOut);

}

INSTANTIATE_CLASS(DropOutLayer);
