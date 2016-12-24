#include"poolLayer.hpp"
#include"common/cudnn.hpp"
#include"test/test.hpp"

#include<glog/logging.h>

/*
 * Create CUDNN handles
 */
template<typename Ntype>
void PoolLayer<Ntype>:: createHandles()
{
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&srcTensorDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dstTensorDesc));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
}

/*
 * Destroy CUDNN Handles
 */
template<typename Ntype>
void PoolLayer<Ntype>:: destroyHandles()
{
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(srcTensorDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dstTensorDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(srcDiffTensorDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dstDiffTensorDesc));
}

template<typename Ntype>
PoolLayer<Ntype>::~PoolLayer()
{
    delete this->m_top;
    destroyHandles();
}

template<typename Ntype>
void PoolLayer<Ntype>::ReShape()
{
    this->m_top = new NDMatrix<Ntype>(this->m_number, this->m_channels, this->m_height, this->m_width);
}

/*
 * Pool layer constructor
 */
template<typename Ntype>
PoolLayer<Ntype>::PoolLayer(string name)
{
    this->m_name = name;
    this->m_inputName = " ";
    this->m_bottom = NULL;
    this->m_top = NULL;
    this->m_loss = 0;
    this->m_prevLayer.clear();
    this->m_nextLayer.clear();

    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    poolingDesc = NULL;
    srcDiffTensorDesc = NULL;
    dstDiffTensorDesc = NULL;

    PoolLayerConfig* curConfig = (PoolLayerConfig*) ConfigTable::getInstance()->getLayerByName(this->m_name);
    string prevLayerName = curConfig->getInput();
    Layer<Ntype>* prev_Layer = (Layer<Ntype>*) LayerContainer<Ntype>::getInstance()->getLayerByName(prevLayerName);

    poolingMode = (cudnnPoolingMode_t)curConfig->getPoolType();
    m_poolDim = curConfig->getSize();
    m_pad_h = curConfig->getPad_h();
    m_pad_w = curConfig->getPad_w();
    m_stride_h =  curConfig->getStride_h();
    m_stride_w = curConfig->getStride_w();
    CHECK_EQ(m_pad_h, m_pad_w);
    CHECK_EQ(m_stride_h, m_stride_w);

    this->m_bottom = prev_Layer->getTop();
    m_prev_num = this->m_bottom->ND_num();
    m_prev_channels = this->m_bottom->ND_channels();
    m_prev_height = this->m_bottom->ND_height();
    m_prev_width = this->m_bottom->ND_width();

    this->m_inputChannels = this->m_bottom->ND_channels();
    this->m_number = m_prev_num;
    this->m_channels = m_prev_channels;
    this->m_height = static_cast<int>(ceil(static_cast<float>(m_prev_height + 2 * m_pad_h - m_poolDim) / m_stride_h)) + 1 ;
    this->m_width = static_cast<int>(ceil(static_cast<float>(m_prev_width + 2 * m_pad_w - m_poolDim) / m_stride_w)) + 1 ;

    ReShape();
    this->createHandles();
}

/*
 * Deep copy constructor
 */
//PoolLayer::PoolLayer(const PoolLayer* layer)
//{
//    srcData = NULL;
//    dstData = NULL;
//    diffData = NULL;
//    m_poolMethod = NULL;
//    prevLayer.clear();
//    nextLayer.clear();
//    srcTensorDesc = NULL;
//    dstTensorDesc = NULL;
//    poolingDesc = NULL;
//    srcDiffTensorDesc = NULL;
//    dstDiffTensorDesc = NULL;
//
//
//    static int idx = 0;
//    _name = layer->_name + string("_") + int_to_string(idx);
//    idx ++;
//    _inputName = layer->_inputName;
//    PoolingMode = layer->PoolingMode;
//    poolDim = layer->poolDim;
//    pad_h = layer->pad_h;
//    pad_w = layer->pad_w;
//    stride_h = layer->stride_h;
//    stride_w = layer->stride_w;
//
//    prev_num = layer->prev_num;
//    prev_channels = layer->prev_channels;
//    prev_height = layer->prev_height;
//    prev_width = layer->prev_width;
//
//    inputImageDim = layer->inputImageDim;
//    inputAmount = layer->inputAmount;
//    number = layer->number;
//    channels = layer->channels;
//    height = layer->height;
//    width = layer->width;
//    outputSize = layer->outputSize;
//
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &diffData, prev_num * prev_channels * prev_height * prev_width * sizeof(float));
//
//    this->createHandles();
//    cout<<"Pool-copy"<<endl;
//}

/*
 * Pool layer Forward propagation
 */
template<typename Ntype>
Ntype PoolLayer<Ntype>::Forward(Phase Phase)
{
    this->m_bottom = this->m_prevLayer[0]->getTop();

    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc,
                                           poolingMode,
                                           CUDNN_PROPAGATE_NAN,
                                           m_poolDim,
                                           m_poolDim,//window
                                           m_pad_h,
                                           m_pad_w,//pading
                                           m_stride_h,
                                           m_stride_w));//stride

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                          cuDNN<float>::getInstance()->GetTensorFormat(),
                                          cuDNN<float>::getInstance()->GetDataType(),
                                          m_prev_num,
                                          m_prev_channels,
                                          m_prev_height,
                                          m_prev_width));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                          cuDNN<float>::getInstance()->GetTensorFormat(),
                                          cuDNN<float>::getInstance()->GetDataType(),
                                          this->m_number,
                                          this->m_channels,
                                          this->m_height,
                                          this->m_width));

    float alpha = 1.0;
    float beta = 0.0;
    CUDNN_CHECK(cudnnPoolingForward(cuDNN<float>::getInstance()->GetcudnnHandle(),
                                   poolingDesc,
                                   &alpha,
                                   srcTensorDesc,
                                   this->m_bottom->gpu_data(),
                                   &beta,
                                   dstTensorDesc,
                                   this->m_top->mutable_gpu_data()));
}

/*
 * Pool layer Backward propagation
 */
template<typename Ntype>
void PoolLayer<Ntype>::Backward()
{
//    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dstTensorDesc,
//                                          cuDNN<float>::getInstance()->GetTensorFormat(),
//                                          cuDNN<float>::getInstance()->GetDataType(),
//                                          number,
//                                          channels,
//                                          height,
//                                          width));
//
//    CUDNN_CHECK(cudnnSetTensor4dDescriptor(srcDiffTensorDesc,
//                                          cuDNN<float>::getInstance()->GetTensorFormat(),
//                                          cuDNN<float>::getInstance()->GetDataType(),
//                                          number,
//                                          channels,
//                                          height,
//                                          width));
//
//    CUDNN_CHECK(cudnnSetTensor4dDescriptor(srcTensorDesc,
//                                          cuDNN<float>::getInstance()->GetTensorFormat(),
//                                          cuDNN<float>::getInstance()->GetDataType(),
//                                          prev_num,
//                                          prev_channels,
//                                          prev_height,
//                                          prev_width));
//
//    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dstDiffTensorDesc,
//                                          cuDNN<float>::getInstance()->GetTensorFormat(),
//                                          cuDNN<float>::getInstance()->GetDataType(),
//                                          prev_num,
//                                          prev_channels,
//                                          prev_height,
//                                          prev_width));
//
//    float alpha = 1.0f;
//    float beta = 0.0;
//    int nIndex = m_nCurBranchIndex;
//    CUDNN_CHECK(cudnnPoolingBackward(cuDNN<float>::getInstance()->GetcudnnHandle(),
//                                    poolingDesc,
//                                    &alpha,
//                                    dstTensorDesc,
//                                    dstData,
//                                    srcDiffTensorDesc,
//                                    nextLayer[nIndex]->diffData,
//                                    srcTensorDesc,
//                                    srcData,
//                                    &beta,
//                                    dstDiffTensorDesc,
//                                    diffData));
}

INSTANTIATE_CLASS(PoolLayer);
