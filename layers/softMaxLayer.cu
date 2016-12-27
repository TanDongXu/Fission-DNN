#include"softMaxLayer.hpp"
#include<glog/logging.h>
#include"common/syncedmem.hpp"
#include"layers/dataLayer.hpp"
#include"test/test.hpp"

/*
 * Create CUDNN Handles
 */
template<typename Ntype>
void SoftMaxLayer<Ntype>::createHandles()
{
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_tensorDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_tensorDesc));
}

/*
 * Destroy the CUDNN Handles
 */
template<typename Ntype>
void SoftMaxLayer<Ntype>:: destroyHandles()
{
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_tensorDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_tensorDesc));
}

/*
 * Destructor
 */
template<typename Ntype>
SoftMaxLayer<Ntype>::~SoftMaxLayer()
{
    delete this->m_top;
    destroyHandles();
}

/*
 * Get the datasize and label
 */
template<typename Ntype>
void SoftMaxLayer<Ntype>::getBatch_labels()
{
    DataLayer<Ntype>* data_Layer = (DataLayer<Ntype>*) LayerContainer<Ntype>::getInstance()->getLayerByName("data");
    m_dataSize = data_Layer->getDataSize();
    m_srcLabels = data_Layer->getLabels();
}

template<typename Ntype>
void SoftMaxLayer<Ntype>::ReShape()
{
    this->m_top = new NDMatrix<Ntype>(this->m_number, this->m_channels, this->m_height, this->m_width);
}

/*
 * SoftMax layer constructor
 */
template<typename Ntype>
SoftMaxLayer<Ntype>::SoftMaxLayer(string name)
{
    this->m_name = name;
    this->m_inputName = " ";
    this->m_bottom = NULL;
    this->m_top = NULL;
    this->m_loss = 0;
    this->m_prevLayer.clear();
    this->m_nextLayer.clear();

    m_srcLabels = NULL;
    bottom_tensorDesc = NULL;
    top_tensorDesc = NULL;

    m_flag = 0;
    m_dataSize = 0;
    m_correctSize = 0;
    m_curCorrectSize = 0;

    SoftMaxLayerConfig* curConfig = (SoftMaxLayerConfig*) ConfigTable::getInstance()->getLayerByName(this->m_name);
    string prevLayerName = curConfig->getInput();
    Layer<Ntype>* prev_Layer =(Layer<Ntype>*) LayerContainer<Ntype>::getInstance()->getLayerByName(prevLayerName);

    m_batchSize = ConfigTable::getInstance()->getBatchSize();
    m_nClasses = curConfig->getNClasses();

    this->m_bottom = prev_Layer->getTop();
    CHECK(this->m_bottom);
    this->m_inputChannels = this->m_bottom->ND_channels();
    this->m_number = this->m_bottom->ND_num();
    this->m_channels = this->m_bottom->ND_channels();
    this->m_height = this->m_bottom->ND_height();
    this->m_width = this->m_bottom->ND_width();
    CHECK_EQ(this->m_height, 1);
    CHECK_EQ(this->m_width, 1);

    this->createHandles();
    ReShape();
}

/*
 * Deep copy constructor
 * */
//SoftMaxLayer::SoftMaxLayer(const SoftMaxLayer* layer)
//{
//    srcData = NULL;
//    dstData = NULL;
//    srcDiff = NULL;
//    diffData = NULL;
//    devLabel = NULL;
//    srcDiff = NULL;
//    host_result = NULL;
//    srcLabel = NULL;
//    bottom_tensorDesc = NULL;
//    top_tensorDesc = NULL;
//    srcDiffTensorDesc = NULL;
//    dstDiffTensorDesc = NULL;
//    nextLayer.clear();
//    prevLayer.clear();
//    flag = 0;
//    dataSize = 0;
//    CorrectSize = 0;
//    cur_correctSize = 0;
//
//    static int idx = 0;
//    _name = layer->_name + string("_") + int_to_string(idx);
//    idx ++;
//    _inputName = layer->_inputName;
//    batchSize = layer->batchSize;
//    inputSize = layer->inputSize;
//    nclasses = layer->nclasses;
//    lambda = layer->lambda;
//    outputSize = layer->outputSize;
//
//    inputAmount = layer->inputAmount;
//    inputImageDim = layer->inputImageDim;
//    number = layer->number;
//    channels = layer->channels;
//    height = layer->height;
//    width = layer->width;
//
//    host_result = (float*) MemoryMonitor::getInstance()->cpuMallocMemory(number * channels * height * width * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &srcDiff, number * channels * height * width * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &devLabel, batchSize * 1 * 1 * 1 * sizeof(int));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &diffData, number * channels * height * width * sizeof(float));
//    this->createHandles();
//}

/*
 * Classification results
 */
template<typename Ntype>
void SoftMaxLayer<Ntype>::ClassificationResults()
{
    if(m_flag == 0)
    {
        m_curCorrectSize = m_dataSize;
    }

    const int max_digit = m_nClasses;

    int temp = ((this->m_number <= m_dataSize - m_flag) ? this->m_number : (m_dataSize - m_flag));
    for(int i = 0; i < temp; i++)
    {
        float max = this->m_top->cpu_data()[i * max_digit];
        int labelIndex = 0;
        for(int j = 1; j < max_digit; j++)
        {
            if(max < this->m_top->cpu_data()[i * max_digit + j])
            {
                max = this->m_top->cpu_data()[i * max_digit + j];
                labelIndex = j;
            }
        }

        m_flag++;
        if(m_srcLabels->cpu_data()[i] != labelIndex) --m_curCorrectSize;
    }

    if(m_flag == m_dataSize)
    {
        cout<< this->m_name << " " << m_curCorrectSize << "/" << m_correctSize <<" ";
        if(m_curCorrectSize > m_correctSize)
        {
            m_correctSize = m_curCorrectSize;
        }
        m_flag = 0;
    }
}

/*
 * Softmax layer forward propagation
 */
template<typename Ntype>
void SoftMaxLayer<Ntype>::Forward(Phase phase)
{
    getBatch_labels();
    this->m_bottom = this->m_prevLayer[0]->getTop();

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
    CUDNN_CHECK(cudnnSoftmaxForward(cuDNN<float>::getInstance()->GetcudnnHandle(),
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &alpha,
                                   bottom_tensorDesc,
                                   this->m_bottom->gpu_data(),
                                   &beta,
                                   top_tensorDesc,
                                   this->m_top->mutable_gpu_data()));

    // If test, compute result
    if(TEST == phase ) ClassificationResults();
}

/*
 * Compute the diff
 */
__global__ void SoftmaxLossBackprop(const int* label, int num_labels, int batch_size, float* diffData)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const int label_value = label[idx];
    /* For each item in the batch, decrease the result of the label's value by 1*/
    diffData[idx * num_labels + label_value] -= 1.0f;
}


/*
 * Compute diff
 */
template<typename Ntype>
void SoftMaxLayer<Ntype>::getBackPropDiffData()
{
    mem_gpu2gpu(this->m_top->mutable_gpu_diff(), this->m_top->mutable_gpu_data(), this->m_number * this->m_channels * this->m_height * this->m_width * sizeof(float));
    SoftmaxLossBackprop<<< (m_batchSize + 127)/128, 128>>>(m_srcLabels->gpu_data(), m_nClasses, m_batchSize, (float*)this->m_top->mutable_gpu_diff());
    cudaThreadSynchronize();
}

/*
 * SoftMAX backward propagation
 */
template<typename Ntype>
void SoftMaxLayer<Ntype>::Backward()
{
    getBackPropDiffData();
    float alpha = 1.0f;
    float beta = 0.0f;
    /*
     * Computes the gridient of the softmax
     */
    CUDNN_CHECK(cudnnSoftmaxBackward(cuDNN<float>::getInstance()->GetcudnnHandle(),
                                    CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &alpha,
                                    top_tensorDesc,
                                    this->m_top->gpu_data(),
                                    top_tensorDesc,
                                    this->m_top->gpu_diff(),
                                    &beta,
                                    bottom_tensorDesc,
                                    this->m_bottom->mutable_gpu_diff()));
}


INSTANTIATE_CLASS(SoftMaxLayer);
