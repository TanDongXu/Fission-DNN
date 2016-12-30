#include"convLayer.hpp"
#include"config/configBase.hpp"
#include"common/util/util.cuh"
#include"common/syncedmem.hpp"
#include"common/cudnn.hpp"
#include"common/common.hpp"
#include"test/test.hpp"

#include<cuda_runtime_api.h>
#include<glog/logging.h>

/*
 * Create handles
 */
template<typename Ntype>
void ConvLayer<Ntype>::createHandles()
{
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_tensorDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_tensorDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&biasTensorDesc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    curandCreateGenerator(&curandGenerator_W, CURAND_RNG_PSEUDO_MTGP32);
    curandCreateGenerator(&curandGenerator_B, CURAND_RNG_PSEUDO_MTGP32);
}

/*
 * Destroy the handles
 */
template<typename Ntype>
void ConvLayer<Ntype>:: destroyHandles()
{
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_tensorDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_tensorDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(biasTensorDesc));
    curandDestroyGenerator(curandGenerator_W);
   	curandDestroyGenerator(curandGenerator_B);
}

/*
 * Random initial weights and Bias
 */
template<typename Ntype>
void ConvLayer<Ntype>::initRandom(bool isGaussian)
{
    //srand((unsigned)time(NULL));
    if(isGaussian)
    {
        createGaussian<Ntype>(m_weight, m_epsilon);
    }else
    {
        //set seed
        curandSetPseudoRandomGeneratorSeed(curandGenerator_W, time(NULL));
        curandGenerateNormal(curandGenerator_W, (float*)m_weight->mutable_gpu_data(), m_kernelAmount * this->m_inputChannels * m_kernelSize * m_kernelSize, 0, m_epsilon);
    }
    // memset bias
    gpuMemoryMemset(m_bias->mutable_gpu_data(), m_kernelAmount * 1 * 1 * 1 * sizeof(Ntype));
}

template<typename Ntype>
void ConvLayer<Ntype>::ReShape()
{
    this->m_top = new NDMatrix<Ntype>(this->m_number, this->m_channels, this->m_height, this->m_width);
    m_weight = new NDMatrix<Ntype>(m_kernelAmount, this->m_inputChannels, m_kernelSize, m_kernelSize);
    m_bias = new NDMatrix<Ntype>(m_kernelAmount, 1, 1, 1);
}

/*
 * ConvLayer constructor
 */
template<typename Ntype>
ConvLayer<Ntype>::ConvLayer(string name)
{
    this->m_name = name;
    this->m_inputName = " ";
    this->m_bottom = NULL;
    this->m_top = NULL;
    this->m_loss = 0;
    this->m_prevLayer.clear();
    this->m_nextLayer.clear();
    m_weight = NULL;
    m_bias = NULL;
    tmp_Wgrad = NULL;
    tmp_Bgrad = NULL;

    filterDesc = NULL;
    convDesc = NULL;
    bottom_tensorDesc = NULL;
    top_tensorDesc = NULL;
    biasTensorDesc = NULL;
    convFwdAlgo = (cudnnConvolutionFwdAlgo_t)-1;
    convBwdFilterAlgo = (cudnnConvolutionBwdFilterAlgo_t)-1;
    convBwdDataAlgo = (cudnnConvolutionBwdDataAlgo_t)-1;

    m_batchSize = ConfigTable::getInstance()->getBatchSize();
    m_momentum = ConfigTable::getInstance()->getMomentum();
    ConvLayerConfig* curConfig = (ConvLayerConfig*) ConfigTable::getInstance()->getLayerByName(this->m_name);
    bool isGaussian = curConfig->isGaussian();
    string prevLayerName = curConfig->getInput();
    Layer<Ntype>* prev_layer = (Layer<Ntype>*) LayerContainer<Ntype>::getInstance()->getLayerByName(prevLayerName);

    m_epsilon = curConfig->getInit_w();
    this->m_lrate = curConfig->getLrate();;
    m_kernelAmount = curConfig->getKernelAmount();
    m_kernelSize = curConfig->getKernelSize();
    m_pad_h = curConfig->getPad_h();
    m_pad_w = curConfig->getPad_w();
    m_stride_h = curConfig->getStride_h();
    m_stride_w = curConfig->getStride_w();
    m_lambda = curConfig->getWeight_decay();
    CHECK_EQ(m_pad_h, m_pad_w);
    CHECK_EQ(m_stride_h, m_stride_w);

    this->m_bottom = prev_layer->getTop();
    CHECK(this->m_bottom);
    this->m_inputChannels = this->m_bottom->ND_channels();
    m_prev_num = this->m_bottom->ND_num();
    m_prev_channels = this->m_inputChannels;
    m_prev_height = this->m_bottom->ND_height();
    m_prev_width = this->m_bottom->ND_width();
    this->m_number = m_prev_num;
    this->m_channels = m_kernelAmount;
    this->m_height = (m_prev_height + 2 * m_pad_h - m_kernelSize) / m_stride_h + 1;
    this->m_width = (m_prev_width + 2 * m_pad_w - m_kernelSize) / m_stride_w + 1;
    CHECK_EQ(this->m_height, this->m_width);
    CHECK_EQ(this->m_number, m_batchSize);
    
    mallocDeviceMem((void**)&tmp_Wgrad, this->m_inputChannels * m_kernelAmount * m_kernelSize * m_kernelSize * sizeof(float));
    mallocDeviceMem((void**)&tmp_Bgrad, m_kernelAmount * 1 * 1 * 1 * sizeof(float));
    // reShape the weight, bias, top , bottom
    this->createHandles();
    ReShape();
    this->initRandom(isGaussian);
}

///*
// * Deep copy constructor for convolution layers
// */
//ConvLayer::ConvLayer(const ConvLayer* layer)
//{
//    srcData = NULL;
//    dstData = NULL;
//    host_Weight = NULL;
//    host_Bias = NULL;
//    dev_Weight = NULL;
//    dev_Bias = NULL;
//    dev_Wgrad = NULL;
//    dev_Bgrad = NULL;
//    tmp_Wgrad = NULL;
//    tmp_Bgrad = NULL;
//    diffData = NULL;
//    prevLayer.clear();
//    nextLayer.clear();
//
//    filterDesc = NULL;
//    convDesc = NULL;
//    bottom_tensorDesc = NULL;
//    top_tensorDesc = NULL;
//    biasTensorDesc = NULL;
//    srcDiffTensorDesc = NULL;
//    dstDiffTensorDesc = NULL;
//    convFwdAlgo = (cudnnConvolutionFwdAlgo_t)-1;
//    convBwdFilterAlgo = (cudnnConvolutionBwdFilterAlgo_t)-1;
//    convBwdDataAlgo = (cudnnConvolutionBwdDataAlgo_t)-1;
//
//    static int idx = 0;
//    _name = layer->_name + string("_") + int_to_string(idx);
//    idx ++;
//    _inputName = layer->_inputName ;
//    epsilon = layer->epsilon;
//    lrate = layer->lrate;
//    batchSize = layer->batchSize;
//    kernelAmount = layer->kernelAmount;
//    kernelSize = layer->kernelSize;
//    pad_h = layer->pad_h;
//    pad_w = layer->pad_w;
//    stride_h = layer->stride_h;
//    stride_w = layer->stride_w;
//    lambda = layer->lambda;
//    inputAmount = layer->inputAmount;
//    inputImageDim = layer->inputImageDim;
//    prev_num = layer->prev_num;
//    prev_channels = layer->prev_channels;
//    prev_height = layer->prev_height;
//    prev_width = layer->prev_width;
//    number = layer->number;
//    channels = layer->channels;
//    height = layer->height;
//    width = layer->width;
//    outputSize = layer->outputSize;
//
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**)&dev_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**)&dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**)&tmp_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**)&tmp_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**)&dstData, batchSize * kernelAmount * height * width * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMallocMemory((void**)&diffData, batchSize * inputAmount * inputImageDim * inputImageDim * sizeof(float));
//    //    MemoryMonitor::getInstance()->gpu2gpu(dev_Wgrad, layer->dev_Wgrad, kernelAmount * inputAmount * 1 * kernelSize * kernelSize * sizeof(float));
//    //    MemoryMonitor::getInstance()->gpu2gpu(dev_Bgrad, layer->dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMemoryMemset(dev_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
//    MemoryMonitor::getInstance()->gpuMemoryMemset(dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
//    this->createHandles();
//    this->initRandom();
//    cout<<"Conv-copy"<<endl;
//}

/*
 * Destructor
 */
template<typename Ntype>
ConvLayer<Ntype>::~ConvLayer()
{
    destroyHandles();
    delete this->m_top;
    delete this->m_weight;
    delete this->m_bias;
}

/*
 * Forward propagation add Bias
 */
template<typename Ntype>
void ConvLayer<Ntype>::addBias(const cudnnTensorDescriptor_t& top_tensorDesc, int c, Ntype* data )
{

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(biasTensorDesc,
                                          cuDNN<float>::getInstance()->GetTensorFormat(),
                                          cuDNN<float>::getInstance()->GetDataType(),
                                          1,
                                          c,
                                          1,
                                          1));

    float alpha = 1.0;
    float beta = 1.0;
    CUDNN_CHECK(cudnnAddTensor(cuDNN<float>::getInstance()->GetcudnnHandle(),
                              &alpha,
                              biasTensorDesc,
                              m_bias->gpu_data(),
                              &beta,
                              top_tensorDesc,
                              data));
}

/*
 * Convolution forward propagation
 */
template<typename Ntype>
void ConvLayer<Ntype>::Forward(Phase phase)
{
    this->m_bottom = this->m_prevLayer[0]->getTop();

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bottom_tensorDesc,
                                          cuDNN<float>::getInstance()->GetTensorFormat(),
                                          cuDNN<float>::getInstance()->GetDataType(),
                                          m_prev_num,
                                          m_prev_channels,
                                          m_prev_height,
                                          m_prev_width));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc,
                                          cuDNN<float>::getInstance()->GetDataType(),
                                          cuDNN<float>::getInstance()->GetTensorFormat(),
                                          m_kernelAmount,
                                          this->m_inputChannels,
                                          m_kernelSize,
                                          m_kernelSize));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                               m_pad_h,
                                               m_pad_w,//pading
                                               m_stride_h,
                                               m_stride_w,//stride
                                               1,1,//upscale
                                               CUDNN_CROSS_CORRELATION));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(top_tensorDesc,
                                          cuDNN<float>::getInstance()->GetTensorFormat(),
                                          cuDNN<float>::getInstance()->GetDataType(),
                                          this->m_number,
                                          this->m_channels,
                                          this->m_height,
                                          this->m_width));

    /*
     * Obtain the best suited algorithm for cudnnConvolutinForward
     * */
    if (cuDNN<float>::getInstance()->getConvFwdAlgorithm() < 0)
    {
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cuDNN<float>::getInstance()->GetcudnnHandle(),
                                                       bottom_tensorDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       top_tensorDesc,
                                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                       0,
                                                       &convFwdAlgo));

        cuDNN<float>::getInstance()->setConvolutionFwdAlgorithm(convFwdAlgo);
    }else
    {
    	convFwdAlgo =(cudnnConvolutionFwdAlgo_t)cuDNN<float>::getInstance()->getConvFwdAlgorithm();
    }

    /*Get the amount of GPU memory for cudnnConvolutionForward*/
    size_t convFwdSizeInBytes = 0;
    void* convFwdWorkSpace = NULL;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cuDNN<float>::getInstance()->GetcudnnHandle(),
                                                       bottom_tensorDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       top_tensorDesc,
                                                       convFwdAlgo,
                                                       &convFwdSizeInBytes));

    if (convFwdSizeInBytes != 0)
    {
        CUDA_CHECK(cudaMalloc(&convFwdWorkSpace, convFwdSizeInBytes));
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(cuDNN<float>::getInstance()->GetcudnnHandle(),
                                       &alpha,
                                       bottom_tensorDesc,
                                       this->m_bottom->gpu_data(),
                                       filterDesc,
                                       m_weight->gpu_data(),
                                       convDesc,
                                       convFwdAlgo,
                                       convFwdWorkSpace,
                                       convFwdSizeInBytes,
                                       &beta,
                                       top_tensorDesc,
                                       this->m_top->mutable_gpu_data()));

    /*add bias*/
    addBias(top_tensorDesc, this->m_channels,this->m_top->mutable_gpu_data());

    if (convFwdSizeInBytes != 0)
    {
        CUDA_CHECK(cudaFree(convFwdWorkSpace));
    }

}

/*
 * Convolution backward propagation
 */
template<typename Ntype>
void ConvLayer<Ntype>::Backward()
{
    // Get the convolutuion function gradient with respect to the bias
    float alpha = 1.0f;
    float beta = 0.0f;
    //int nIndex = m_nCurBranchIndex;
    CUDNN_CHECK(cudnnConvolutionBackwardBias(cuDNN<float>::getInstance()->GetcudnnHandle(),
                                            &alpha,
                                            top_tensorDesc,
                                            this->m_top->gpu_diff(),
                                            &beta,
                                            biasTensorDesc,
                                            tmp_Bgrad
                                            ));

    // Obtain the best suited algorithm for cudnnConvolutionBackwardFilter
    if(cuDNN<float>::getInstance()->getConvolutionBwdFilterAlgorithm() < 0)
    {
    	CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(cuDNN<float>::getInstance()->GetcudnnHandle(),
    			                                               bottom_tensorDesc,
    			                                               top_tensorDesc,
    			                                               convDesc,
    			                                               filterDesc,
    			                                               CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
    			                                               0,
    			                                               &convBwdFilterAlgo
    			                                               ));

    	cuDNN<float>::getInstance()->setConvolutionBwdFilterAlgorithm(convBwdFilterAlgo);
    }else
    {
    	convBwdFilterAlgo = (cudnnConvolutionBwdFilterAlgo_t)cuDNN<float>::getInstance()->getConvolutionBwdFilterAlgorithm();
    }

    /*Get the GPU memory workspace for cudnnConvolutionBackwardFilter*/
    size_t convBwdFilterSizeInBytes = 0;
    void* convBwdFilterWorkSpace = NULL;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(cuDNN<float>::getInstance()->GetcudnnHandle(),
    		                                                  bottom_tensorDesc,
    		                                                  top_tensorDesc,
    		                                                  convDesc,
    		                                                  filterDesc,
    		                                                  convBwdFilterAlgo,
    		                                                  &convBwdFilterSizeInBytes
    /*Alloc GPU memory*/		                                                  ));
    if(convBwdFilterSizeInBytes != 0)
    {
    	CUDA_CHECK(cudaMalloc(&convBwdFilterWorkSpace, convBwdFilterSizeInBytes));
    }

   /*This function computes the convolution gradient with respect to filter coefficient using the specified algo*/
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(cuDNN<float>::getInstance()->GetcudnnHandle(),
                                              &alpha,
                                              bottom_tensorDesc,
                                              this->m_bottom->gpu_data(),
                                              top_tensorDesc,
                                              this->m_top->gpu_diff(),
                                              convDesc,
                                              convBwdFilterAlgo,
                                              convBwdFilterWorkSpace,
                                              convBwdFilterSizeInBytes,
                                              &beta,
                                              filterDesc,
                                              tmp_Wgrad
                                              ));

    if (convBwdFilterSizeInBytes != 0)
    {
        CUDA_CHECK(cudaFree(convBwdFilterWorkSpace));
    }

    /*Obtaining the best suited algorithm for the cudnnConvolutionBackwardData*/
    if(cuDNN<float>::getInstance()->getConvolutionBwdDataAlgorithm() < 0)
    {
    	CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(cuDNN<float>::getInstance()->GetcudnnHandle(),
    			                                            filterDesc,
    			                                            top_tensorDesc,
    			                                            convDesc,
    			                                            top_tensorDesc,
    			                                            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
    			                                            0,
    			                                            &convBwdDataAlgo
    			                                            ));
    	cuDNN<float>::getInstance()->setConvolutionBwdDataAlgorithm(convBwdDataAlgo);

    }else
    {
    	convBwdDataAlgo = (cudnnConvolutionBwdDataAlgo_t)cuDNN<float>::getInstance()->getConvolutionBwdDataAlgorithm();
    }

    /*Get the amount of GPU memory for the cudnnConvlotionBackwardData*/
    size_t convBwdDataSizeInBytes = 0;
    void* convBwdDataWorkSpace = NULL;
    /*按照接口说明bottom_tensorDesc应该是top_tensorDesc的,参考一个代码是用bottom_tensorDesc*/
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(cuDNN<float>::getInstance()->GetcudnnHandle(),
    		                                                filterDesc,
    		                                                top_tensorDesc,
    		                                                convDesc,
    		                                                bottom_tensorDesc,
    		                                                convBwdDataAlgo,
    		                                                &convBwdDataSizeInBytes
    		                                                ));
    if(convBwdDataSizeInBytes != 0)
    {
    	CUDA_CHECK(cudaMalloc(&convBwdDataWorkSpace, convBwdDataSizeInBytes));
    }

    //Note:if use convBwdDataAlgo above,it will return error in running.
    // convBwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    /*Compute the convolution gradient with respect to the output tensor using the specified algo*/
    alpha = 1.0f;
    beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(cuDNN<float>::getInstance()->GetcudnnHandle(),
                                            &alpha,
                                            filterDesc,
                                            m_weight->gpu_data(),
                                            top_tensorDesc,
                                            this->m_top->gpu_diff(),
                                            convDesc,
                                            convBwdDataAlgo,
                                            convBwdDataWorkSpace,
                                            convBwdDataSizeInBytes,
                                            &beta,
                                            bottom_tensorDesc,
                                            this->m_bottom->mutable_gpu_diff()));

    if(convBwdDataSizeInBytes != 0)
    {
    	CUDA_CHECK(cudaFree(convBwdDataWorkSpace));
    }

    /*
     * Update the weights in conv layer
     *
     * */
    alpha = m_lambda * m_batchSize;
    int size =  m_kernelAmount * this->m_inputChannels * m_kernelSize * m_kernelSize;
    CUBLAS_CHECK(cublasSaxpy(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  size,
                                  &alpha,
                                  (float*)m_weight->mutable_gpu_data(),
                                  1,
                                  tmp_Wgrad,
                                  1));

    float scalVal = m_momentum;
    size =  m_kernelAmount * this->m_inputChannels * m_kernelSize * m_kernelSize;
    CUBLAS_CHECK(cublasSscal(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  size,
                                  &scalVal,
                                  (float*)m_weight->mutable_gpu_diff(),
                                  1));

    size = m_kernelAmount * 1 * 1 * 1;
    CUBLAS_CHECK(cublasSscal(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  size,
                                  &scalVal,
                                  (float*)m_bias->mutable_gpu_diff(),
                                  1));

    scalVal = this->m_lrate * 1.0f / m_batchSize;
    size =  m_kernelAmount * this->m_inputChannels * m_kernelSize * m_kernelSize;
    CUBLAS_CHECK(cublasSaxpy(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  size,
                                  &scalVal,
                                  tmp_Wgrad,
                                  1,
                                  (float*)m_weight->mutable_gpu_diff(),
                                  1));

    scalVal = 2 * this->m_lrate * 1.0f / m_batchSize;
    size = m_kernelAmount * 1 * 1 * 1;
    CUBLAS_CHECK(cublasSaxpy(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  size,
                                  &scalVal,
                                  tmp_Bgrad,
                                  1,
                                  (float*)m_bias->mutable_gpu_diff(),
                                  1));

    alpha = -1.0f;
    size =  m_kernelAmount * this->m_inputChannels * m_kernelSize * m_kernelSize;
    CUBLAS_CHECK(cublasSaxpy(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  size,
                                  &alpha,
                                  (float*)m_weight->mutable_gpu_diff(),
                                  1,
                                  (float*)m_weight->mutable_gpu_data(),
                                  1));

    size = m_kernelAmount * 1 * 1 * 1;
    CUBLAS_CHECK(cublasSaxpy(cuDNN<float>::getInstance()->GetcublasHandle(),
                                  size,
                                  &alpha,
                                  (float*)m_bias->mutable_gpu_diff(),
                                  1,
                                  (float*)m_bias->mutable_gpu_data(),
                                  1));
}



INSTANTIATE_CLASS(ConvLayer);
