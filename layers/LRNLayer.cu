#include"LRNLayer.hpp"
#include"common/common.hpp"
#include"config/configBase.hpp"
#include"common/cudnn.hpp"
#include"test/test.hpp"

/*
 * Create CUDNN Handles
 */
template<typename Ntype>
void LRNLayer<Ntype>::createHandles()
{
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_tensorDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_tensorDesc));
    CUDNN_CHECK(cudnnCreateLRNDescriptor(&normDesc));
}

/*
 * Destroy CUDNN Handles
 */
template<typename Ntype>
void LRNLayer<Ntype>::destroyHandles()
{
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_tensorDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_tensorDesc));
    CUDNN_CHECK(cudnnDestroyLRNDescriptor(normDesc));
}

template<typename Ntype>
void LRNLayer<Ntype>::ReShape()
{
    this->m_top = new NDMatrix<Ntype>(this->m_number, this->m_channels, this->m_height, this->m_width);
}

/*
 * LRN layer constructor
 */
template<typename Ntype>
LRNLayer<Ntype>::LRNLayer(string name)
{
    this->m_name = name;
    this->m_inputName = " ";
    this->m_bottom = NULL;
    this->m_top = NULL;
    this->m_loss = 0;
    this->m_prevLayer.clear();
    this->m_nextLayer.clear();
    bottom_tensorDesc = NULL;
    top_tensorDesc = NULL;

    LRNLayerConfig* curConfig = (LRNLayerConfig*)ConfigTable::getInstance()->getLayerByName(this->m_name);
    string prevLayerName = curConfig->getInput();
    Layer<Ntype>* prev_Layer = (Layer<Ntype>*)LayerContainer<Ntype>::getInstance()->getLayerByName(prevLayerName);

    lrnN = curConfig->getLrnN();
    lrnAlpha = curConfig->getLrnAlpha();
    lrnBeta = curConfig->getLrnBeta();
    lrnK = 1.0;

    this->m_bottom = prev_Layer->getTop();
    CHECK(this->m_bottom);
    this->m_inputChannels = this->m_bottom->ND_channels();
    this->m_number = this->m_bottom->ND_num();
    this->m_channels = this->m_bottom->ND_channels();
    this->m_height = this->m_bottom->ND_height();
    this->m_width = this->m_bottom->ND_width();

    ReShape();
    this->createHandles();
}

/*
 * Deep copy constructor
 */
//LRNLayer::LRNLayer(const LRNLayer* layer)
//{
//    srcData = NULL;
//    dstData = NULL;
//    diffData = NULL;
//    prevLayer.clear();
//    nextLayer.clear();
//    bottom_tensorDesc = NULL;
//    top_tensorDesc = NULL;
//    srcDiffTensorDesc = NULL;
//    dstDiffTensorDesc = NULL;
//
//    static int idx = 0;
//    _name = layer->_name + string("_") + int_to_string(idx);
//    idx ++;
//    _inputName = layer->_inputName;
//
//    lrnN = layer->lrnN;
//    lrnAlpha = layer->lrnAlpha;
//    lrnBeta = layer->lrnBeta;
//    lrnK = layer->lrnK;
//
//    inputAmount = layer->inputAmount;
//    inputImageDim = layer->inputImageDim;
//    number = layer->number;
//    channels = layer->channels;
//    height = layer->height;
//    width = layer->width;
//    inputSize = layer->inputSize;
//    outputSize = layer->outputSize;
//
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &diffData, number * channels * height * width * sizeof(float));
//
//    this->createHandles();
//}
//
/*
 * Destructor
 */
template<typename Ntype>
LRNLayer<Ntype>::~LRNLayer()
{
    delete this->m_top;
    destroyHandles();
}

/*
 * LRN Forward propagation
 */
template<typename Ntype>
Ntype LRNLayer<Ntype>::Forward(Phase phase)
{
    this->m_bottom = this->m_prevLayer[0]->getTop();

    CUDNN_CHECK(cudnnSetLRNDescriptor(normDesc,
                                     lrnN,
                                     lrnAlpha,
                                     lrnBeta,
                                     lrnK));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bottom_tensorDesc,
                                          cuDNN<float>::getInstance()->GetTensorFormat(),
                                          cuDNN<float>::getInstance()->GetDataType(),
                                          this->m_number,
                                          this->m_channels,
                                          this->m_height,
                                          this->m_width));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(top_tensorDesc,
                                          cuDNN<float>::getInstance()->GetTensorFormat(),
                                          cuDNN<float>::getInstance()->GetDataType(),
                                          this->m_number,
                                          this->m_channels,
                                          this->m_height,
                                          this->m_width));

    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CHECK(cudnnLRNCrossChannelForward(cuDNN<float>::getInstance()->GetcudnnHandle(),
                                           normDesc,
                                           CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                           &alpha,
                                           bottom_tensorDesc,
                                           this->m_bottom->gpu_data(),
                                           &beta,
                                           top_tensorDesc,
                                           this->m_top->mutable_gpu_data()));
}

/*
 * LRN Backward propagation
 */
template<typename Ntype>
void LRNLayer<Ntype>::Backward()
{
    float alpha = 1.0f;
    float beta = 0.0f;
    //int nIndex = m_nCurBranchIndex;
    CUDNN_CHECK(cudnnLRNCrossChannelBackward(cuDNN<float>::getInstance()->GetcudnnHandle(),
                                            normDesc,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &alpha,
                                            top_tensorDesc,
                                            this->m_top->gpu_data(),
                                            top_tensorDesc,
                                            this->m_top->gpu_diff(),
                                            bottom_tensorDesc,
                                            this->m_bottom->gpu_data(),
                                            &beta,
                                            bottom_tensorDesc,
                                            this->m_bottom->mutable_gpu_diff()));
}

INSTANTIATE_CLASS(LRNLayer);
