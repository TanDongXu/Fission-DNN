#include"activationLayer.hpp"
#include"config/configBase.hpp"
#include"common/cudnn.hpp"
#include"common/common.hpp"
#include"test/test.hpp"

/*
 * Create CUDNN handles
 */
template<typename Ntype>
void ActivationLayer<Ntype>::createHandles()
{
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&srcTensorDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dstTensorDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activDesc));
}

/*
 * Destroy CUDNN Handles
 */
template<typename Ntype>
void ActivationLayer<Ntype>::destroyHandles()
{
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(srcTensorDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dstTensorDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(srcDiffTensorDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dstDiffTensorDesc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(activDesc));
}

template<typename Ntype>
void ActivationLayer<Ntype>::ReShape()
{
    this->m_top = new NDMatrix<Ntype>(this->m_number, this->m_channels, this->m_height, this->m_width);
}

/*
 * Activation layer constructor
 */
template<typename Ntype>
ActivationLayer<Ntype>::ActivationLayer(string name)
{
    this->m_name = name;
    this->m_inputName = " ";
    this->m_bottom = NULL;
    this->m_top = NULL;
    this->m_loss = 0;
    this->m_prevLayer.clear();
    this->m_nextLayer.clear();
    
    activDesc = NULL;
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    srcDiffTensorDesc = NULL;
    dstDiffTensorDesc = NULL;

    ActivationLayerConfig * curConfig = (ActivationLayerConfig*) ConfigTable::getInstance()->getLayerByName(this->m_name);
    string preLayerName = curConfig->getInput();
    Layer<Ntype>* prev_Layer = (Layer<Ntype>*) LayerContainer<Ntype>::getInstance()->getLayerByName(preLayerName);

    this->m_bottom = prev_Layer->getTop();
    this->m_inputChannels = this->m_bottom->ND_channels();
    this->m_number = this->m_bottom->ND_num();
    this->m_channels = this->m_bottom->ND_channels();
    this->m_height = this->m_bottom->ND_height();
    this->m_width = this->m_bottom->ND_width();
    ActivationMode = curConfig->getNonLinearType();

    ReShape();
    this->createHandles();
}

/*
 * Deep copy constructor
 */
//ActivationLayer::ActivationLayer(const ActivationLayer* layer)
//{
//    srcData = NULL;
//    dstData = NULL;
//    diffData = NULL;
//    prevLayer.clear();
//    nextLayer.clear();
//    activDesc = NULL;
//    srcTensorDesc = NULL;
//    dstTensorDesc = NULL;
//    srcDiffTensorDesc = NULL;
//    dstDiffTensorDesc = NULL;
//
//    static int idx = 0;
//    _name = layer->_name + string("_") + int_to_string(idx);
//    idx ++;
//    _inputName = layer->_inputName;
//    inputAmount = layer->inputAmount;
//    inputImageDim = layer->inputImageDim;
//    number = layer->number;
//    channels =  layer->channels;
//    height = layer->height;
//    width = layer->width;
//    outputSize = layer->outputSize;
//    ActivationMode = layer->ActivationMode;
//
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**)&diffData, number * channels * height * width * sizeof(float));
//
//    this->createHandles();
//    cout<<"Activation-copy"<<endl;
//}

/*
 * Destructor
 */
template<typename Ntype>
ActivationLayer<Ntype>::~ActivationLayer()
{
	delete this->m_top;
    destroyHandles();
}

/*
 * LRELU activation function forward compute
*/
__global__ void LreluForward(const float* srcData, float* dstData, int data_size)
{
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for(int i = 0; i < data_size; i += num_threads)
    {
        int index = i + thread_index;
        if(index < data_size)
        {
            dstData[index] = srcData[index] > 0 ? srcData[index] : srcData[index] * 0.01;
        }
    }

}

/*
 * Activation forward propagation
 */
template<typename Ntype>
Ntype ActivationLayer<Ntype>::Forward(Phase Phase)
{
    this->m_bottom = this->m_prevLayer[0]->getTop();

    if(ActivationMode == ACTIVATION_LRELU)
    {
        int data_size = this->m_number * this->m_channels * this->m_height * this->m_width;
        int num_threads = 256;
        int num_block = (data_size + num_threads - 1) / num_threads;

        LreluForward<<<num_block, num_threads>>>((float*)this->m_bottom->gpu_data(), (float*)this->m_top->mutable_gpu_data(), data_size);
        cudaThreadSynchronize();
    }
    else
    {
        cudnnActivationMode = (cudnnActivationMode_t)ActivationMode;
        CUDNN_CHECK(cudnnSetActivationDescriptor(activDesc,
        		                                cudnnActivationMode,
        		                                CUDNN_PROPAGATE_NAN,
        		                                0.0));

        CUDNN_CHECK(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                              cuDNN<float>::getInstance()->GetTensorFormat(),
                                              cuDNN<float>::getInstance()->GetDataType(),
                                              this->m_number,
                                              this->m_channels,
                                              this->m_height,
                                              this->m_width));

        CUDNN_CHECK(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                              cuDNN<float>::getInstance()->GetTensorFormat(),
                                              cuDNN<float>::getInstance()->GetDataType(),
                                              this->m_number,
                                              this->m_channels,
                                              this->m_height,
                                              this->m_width));

        float alpha = 1.0f;
        float beta = 0.0f;
        CUDNN_CHECK(cudnnActivationForward(cuDNN<float>::getInstance()->GetcudnnHandle(),
                                          activDesc,
                                          &alpha,
                                          srcTensorDesc,
                                          this->m_bottom->gpu_data(),
                                          &beta,
                                          dstTensorDesc,
                                          this->m_top->mutable_gpu_data()));
    }
}

/*
 * LRELU BackWard Compute
*/
__global__ void LreluBackward(float* srcDiff, float* dstDiff, float* srcData, int data_size)
{
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;

    for(int i = 0; i < data_size; i += num_threads)
    {
        int index = i + thread_index;
        if(index < data_size)
        {
            dstDiff[index] = srcDiff[index] * ((srcData[index] > 0) + (srcData[index] <= 0) * 0.01);
        }
    }

}

/*
 * Activation Backward Propagation
 */
template<typename Ntype>
void ActivationLayer<Ntype>::Backward()
{
//    if(ActivationMode == ACTIVATION_LRELU)
//    {
//        int nIndex = m_nCurBranchIndex;
//        int data_size = number * channels * height * width;
//        int num_threads = 256;
//        int num_block = (data_size + num_threads - 1) / num_threads;
//
//        LreluBackward<<<num_block, num_threads>>>(nextLayer[nIndex]->diffData, diffData, srcData, data_size);
//        cudaThreadSynchronize();
//    }
//    else
//    {
//        cudnnActivationMode = (cudnnActivationMode_t)ActivationMode;
//        CUDNN_CHECK(cudnnSetTensor4dDescriptor(dstTensorDesc,
//                                              cuDNN<float>::getInstance()->GetTensorFormat(),
//                                              cuDNN<float>::getInstance()->GetDataType(),
//                                              number,
//                                              channels,
//                                              height,
//                                              width));
//
//        CUDNN_CHECK(cudnnSetTensor4dDescriptor(srcDiffTensorDesc,
//                                              cuDNN<float>::getInstance()->GetTensorFormat(),
//                                              cuDNN<float>::getInstance()->GetDataType(),
//                                              number,
//                                              channels,
//                                              height,
//                                              width));
//
//        CUDNN_CHECK(cudnnSetTensor4dDescriptor(dstDiffTensorDesc,
//                                              cuDNN<float>::getInstance()->GetTensorFormat(),
//                                              cuDNN<float>::getInstance()->GetDataType(),
//                                              number,
//                                              channels,
//                                              height,
//                                              width));
//
//        CUDNN_CHECK(cudnnSetTensor4dDescriptor(srcTensorDesc,
//                                              cuDNN<float>::getInstance()->GetTensorFormat(),
//                                              cuDNN<float>::getInstance()->GetDataType(),
//                                              number,
//                                              channels,
//                                              height,
//                                              width));
//
//        float alpha = 1.0f;
//        float beta = 0.0f;
//        int nIndex = m_nCurBranchIndex;
//        CUDNN_CHECK(cudnnActivationBackward(cuDNN<float>::getInstance()->GetcudnnHandle(),
//                                           activDesc,
//                                           &alpha,
//                                           dstTensorDesc,
//                                           dstData,
//                                           srcDiffTensorDesc,
//                                           nextLayer[nIndex]->diffData,
//                                           srcTensorDesc,
//                                           srcData,
//                                           &beta,
//                                           dstDiffTensorDesc,
//                                           diffData));
//    }
}

INSTANTIATE_CLASS(ActivationLayer);
