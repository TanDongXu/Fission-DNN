#include"hiddenLayer.hpp"
#include"common/syncedmem.hpp"
#include"common/util/util.cuh"
#include"common/common.hpp"
#include"common/cudnn.hpp"

#include<cuda_runtime_api.h>
#include<glog/logging.h>
#include<ctime>
#include<curand.h>

/*
 * Create CUDNN Handles
 */
template<typename Ntype>
void HiddenLayer<Ntype>::createHandles()
{
    curandCreateGenerator(&curandGenerator_W, CURAND_RNG_PSEUDO_MTGP32);
   	curandCreateGenerator(&curandGenerator_B, CURAND_RNG_PSEUDO_MTGP32);
}

/*
 * Destroy CUDNN Handles
 */
template<typename Ntype>
void HiddenLayer<Ntype>::destroyHandles()
{
    curandDestroyGenerator(curandGenerator_W);
   	curandDestroyGenerator(curandGenerator_B);
}

/*
 * Random initial weights and Bias
 */
template<typename Ntype>
void HiddenLayer<Ntype>::initRandom(bool isGaussian)
{
    if(isGaussian)
    {
        createGaussian<Ntype>(m_weight, m_epsilon);
    }else
    {
        srand((unsigned)time(NULL));
   	    /*initial weight*/
   	    curandSetPseudoRandomGeneratorSeed(curandGenerator_W, time(NULL));
        curandGenerateNormal(curandGenerator_W, (float*)m_weight->mutable_gpu_data(), m_outputSize * m_inputSize, 0, m_epsilon);
    }
    //memset bias
    gpuMemoryMemset(m_bias->mutable_gpu_data(), m_outputSize * 1 * 1 * 1 * sizeof(Ntype));
}

/*
 * Fill a float-point array with one
 * */
__global__ void FillOnes(float* vec, int value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > value) return ;

    vec[idx] = 1.0f;
}

template<typename Ntype>
void HiddenLayer<Ntype>::ReShape()
{
    this->m_top = new NDMatrix<Ntype>(m_batchSize, m_outputSize, 1, 1);
    m_weight = new NDMatrix<Ntype>(1, 1, m_outputSize, m_inputSize);
    m_bias = new NDMatrix<Ntype>(m_outputSize, 1, 1, 1);
}

/*
 * Hidden layer constructor
 */
template<typename Ntype>
HiddenLayer<Ntype>::HiddenLayer(string name)
{
    this->m_name = name;
    this->m_inputName = " ";
    this->m_bottom = NULL;
    this->m_top = NULL;
    this->m_loss = 0;
    m_weight = NULL;
    m_bias = NULL;
    VectorOnes = NULL;
    tmp_Wgrad = NULL;
    tmp_Bgrad = NULL;

    this->m_prevLayer.clear();
    this->m_nextLayer.clear();

    m_momentum = ConfigTable::getInstance()->getMomentum();
    HiddenLayerConfig * curConfig = (HiddenLayerConfig*) ConfigTable::getInstance()->getLayerByName(this->m_name);
    string preLayerName = curConfig->getInput();
    Layer<Ntype>* prev_Layer = (Layer<Ntype>*) LayerContainer<Ntype>::getInstance()->getLayerByName(preLayerName);

    m_epsilon = curConfig->getInit_w();
    this->m_lrate = curConfig->getLrate();
    m_outputSize = curConfig->getNumNeurons();;
    m_batchSize = ConfigTable::getInstance()->getBatchSize();
    m_lambda = curConfig->getWeight_decay();
    bool isGaussian = curConfig->isGaussian();

    this->m_bottom = prev_Layer->getTop();
    CHECK(this->m_bottom);
    m_inputSize = this->m_bottom->ND_count(1);
    this->m_inputChannels = this->m_bottom->ND_channels();
    m_prev_num = this->m_bottom->ND_num();
    m_prev_channels = this->m_bottom->ND_channels();
    m_prev_height = this->m_bottom->ND_height();
    m_prev_width = this->m_bottom->ND_width();
    this->m_number = m_prev_num;
    this->m_channels = m_outputSize;
    this->m_height = 1;
    this->m_width = 1;

    //1*batchSize
    mallocDeviceMem((void**)&VectorOnes, 1 * 1 * 1 * m_batchSize* sizeof(float));
    mallocDeviceMem((void**)&tmp_Wgrad, m_outputSize * m_inputSize * sizeof(float));
    mallocDeviceMem((void**)&tmp_Bgrad, m_outputSize * sizeof(float));
    FillOnes<<<1, m_batchSize>>>(VectorOnes, m_batchSize);
    cudaThreadSynchronize();

    this->createHandles();
    ReShape();
    this->initRandom(isGaussian);
}

/*
 * Deep copy constructor
 */
//HiddenLayer::HiddenLayer(const HiddenLayer* layer)
//{
//    srcData = NULL;
//    dstData = NULL;
//    diffData = NULL;
//    host_Weight = NULL;
//    dev_Weight = NULL;
//    host_Bias = NULL;
//    dev_Bias = NULL;
//    dev_Wgrad = NULL;
//    dev_Bgrad = NULL;
//    tmp_Wgrad = NULL;
//    tmp_Bgrad = NULL;
//    VectorOnes = NULL;
//
//    prevLayer.clear();
//    nextLayer.clear();
//
//    static int idx = 0;
//    _name = layer->_name + string("_") + int_to_string(idx);
//    idx ++;
//    _inputName = layer->_inputName;
//    epsilon = layer->epsilon;
//    lrate = layer->lrate;
//    inputSize = layer->inputSize;
//    outputSize = layer->outputSize;
//    batchSize = layer->batchSize;
//    lambda = layer->lambda;
//
//    inputAmount = layer->inputAmount;
//    inputImageDim = layer->inputImageDim;
//    prev_num = layer->prev_num;
//    prev_channels = layer->prev_channels;
//    prev_height = layer->prev_height;
//    prev_width = layer->prev_width;
//    number = layer->number;
//    channels = outputSize;
//    height = 1;
//    width = 1;
//
//    //1*batchSize
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &VectorOnes, 1 * 1 * 1 * batchSize * sizeof(float));
//    FillOnes<<<1, batchSize>>>(VectorOnes, batchSize);
//    cudaThreadSynchronize();
//
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &dev_Wgrad, 1 * 1 * outputSize * inputSize * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &dev_Bgrad, 1 * 1 * outputSize * 1 * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &tmp_Wgrad, 1 * 1 * outputSize * inputSize * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &tmp_Bgrad, 1 * 1 * outputSize * 1 * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &dstData, outputSize * batchSize * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**) &diffData, inputSize * batchSize * sizeof(float));
//
//    //MemoryMonitor::getInstance()->gpu2gpu(dev_Wgrad, layer->dev_Wgrad, 1 * 1 * outputSize * inputSize * sizeof(float));
//    //	MemoryMonitor::getInstance()->gpu2gpu(dev_Bgrad, layer->dev_Bgrad, 1 * 1 * outputSize * 1 * sizeof(float));
//
//    MemoryMonitor::getInstance()->gpuMemoryMemset(dev_Wgrad, 1 * 1 * outputSize * inputSize * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMemoryMemset(dev_Bgrad, 1 * 1 * outputSize * 1 * sizeof(float));
//    
//    this->createHandles();
//    this->initRandom();
//    cout<<"Hidden-copy"<<endl;
//}

/*
 * Destructor
 */
template<typename Ntype>
HiddenLayer<Ntype>::~HiddenLayer()
{
	delete this->m_top;
    delete this->m_weight;
    delete this->m_bias;
    freeDeviceMem(tmp_Wgrad);
    freeDeviceMem(tmp_Bgrad);
    freeDeviceMem(VectorOnes);
	destroyHandles();
}

/*
 * Hidden layer forward propagation
 */
template<typename Ntype>
void HiddenLayer<Ntype>::Forward(Phase phase)
{
    this->m_bottom = this->m_prevLayer[0]->getTop();

    int dim_x = m_prev_channels * m_prev_height * m_prev_width ;
    int dim_y = m_outputSize ;
    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  dim_y,
                                  m_batchSize,
                                  dim_x,
                                  &alpha,
                                  (float*)m_weight->gpu_data(),
                                  dim_x,
                                  (float*)this->m_bottom->gpu_data(),
                                  dim_x,
                                  &beta,
                                  (float*)this->m_top->mutable_gpu_data(),
                                  dim_y));

    //add bias
    alpha = 1.0f;
    beta = 1.0f;
    CUBLAS_CHECK(cublasSgemm(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  dim_y,
                                  m_batchSize,
                                  1,
                                  &alpha,
                                  (float*)m_bias->gpu_data(),
                                  dim_y,
                                  VectorOnes,
                                  1,
                                  &beta,
                                  (float*)this->m_top->mutable_gpu_data(),
                                  dim_y));

    this->m_height = 1; this->m_width = 1; this->m_channels = dim_y;
}

/*
 * Hidden Layer backward propagation
 */
template<typename Ntype>
void HiddenLayer<Ntype>::Backward()
{
    int dim_x = m_prev_channels * m_prev_height * m_prev_width;
    int dim_y = m_outputSize;

    CUDA_CHECK(cudaMemcpy(tmp_Wgrad, m_weight->gpu_data(), 1 * 1 * m_outputSize * m_inputSize * sizeof(float), cudaMemcpyDeviceToDevice));

    float alpha = 1.0f /(float)m_batchSize;
    float beta = m_lambda;
    //int nIndex = m_nCurBranchIndex;
    CUBLAS_CHECK(cublasSgemm(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_T,
                                  dim_x,
                                  dim_y,
                                  m_batchSize,
                                  &alpha,
                                  (float*)this->m_bottom->gpu_data(),
                                  dim_x,
                                  (float*)this->m_top->gpu_diff(),
                                  dim_y,
                                  &beta,
                                  tmp_Wgrad,
                                  dim_x));

    beta = 0.0f;
    CUBLAS_CHECK(cublasSgemv(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  CUBLAS_OP_N,
                                  m_outputSize,
                                  m_batchSize,
                                  &alpha,
                                  (float*)this->m_top->gpu_diff(),
                                  m_outputSize,
                                  VectorOnes,
                                  1,
                                  &beta,
                                  tmp_Bgrad,
                                  1));

    alpha = 1.0f;
    beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  dim_x,
                                  m_batchSize,
                                  m_outputSize,
                                  &alpha,
                                  (float*)m_weight->gpu_data(),
                                  dim_x,
                                  (float*)this->m_top->gpu_diff(),
                                  m_outputSize,
                                  &beta,
                                  (float*)this->m_bottom->mutable_gpu_diff(),
                                  dim_x));

    float scalVal = m_momentum;
    int size = 1 * 1 * m_outputSize * m_inputSize * 1;
    CUBLAS_CHECK(cublasSscal(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  size,
                                  &scalVal,
                                  (float*)m_weight->mutable_gpu_diff(),
                                  1));


    size = 1 * 1 * m_outputSize * 1 * 1;
    CUBLAS_CHECK(cublasSscal(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  size,
                                  &scalVal,
                                  (float*)m_bias->mutable_gpu_diff(),
                                  1));

    scalVal = this->m_lrate;
    size = 1 * 1 * m_outputSize * m_inputSize * 1;
    CUBLAS_CHECK(cublasSaxpy(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  size,
                                  &scalVal,
                                  tmp_Wgrad,
                                  1,
                                  (float*)m_weight->mutable_gpu_diff(),
                                  1));

    scalVal = 2 * this->m_lrate;
    size = m_outputSize * 1 * 1 * 1;
    CUBLAS_CHECK(cublasSaxpy(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  size,
                                  &scalVal,
                                  tmp_Bgrad,
                                  1,
                                  (float*)m_bias->mutable_gpu_diff(),
                                  1));

    /*updata weightt*/
    alpha = -1.0f;
    size = m_outputSize * m_inputSize;
    CUBLAS_CHECK(cublasSaxpy(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  size,
                                  &alpha,
                                  (float*)m_weight->mutable_gpu_diff(),
                                  1,
                                  (float*)m_weight->mutable_gpu_data(),
                                  1));
    size = m_outputSize;
    CUBLAS_CHECK(cublasSaxpy(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  size,
                                  &alpha,
                                  (float*)m_bias->mutable_gpu_diff(),
                                  1,
                                  (float*)m_bias->mutable_gpu_data(),
                                  1));
}


INSTANTIATE_CLASS(HiddenLayer);
