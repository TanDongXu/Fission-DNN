#include"dataLayer.hpp"
#include"common/syncedmem.hpp"
#include"common/common.hpp"
#include"test/test.hpp"
#include"config/configBase.hpp"

#include<cstdlib>
#include<time.h>

/*
 * Datalayer destructor
 */
template<typename Ntype>
DataLayer<Ntype>::~DataLayer()
{
    delete m_data_transformer;
    delete m_labels;
}

template<typename Ntype>
void DataLayer<Ntype>::ReShape()
{
    this->m_bottom = new NDMatrix<Ntype>(m_batchSize, this->m_channels, this->m_height, this->m_width);
    if(m_isDataTransFormer && m_cropSize)
        this->m_top = new NDMatrix<Ntype>(m_batchSize,this->m_channels, m_cropSize, m_cropSize);
    else
    {
        this->m_top = this->m_bottom; 
    }
    // reshape label
    m_labels = new NDMatrix<int>(this->m_number, 1, 1, 1);
    // labels diff no used
    m_labels->diff().reset();

    //pBottom_mutable_cpu_data = this->m_bottom->mutable_cpu_data();
    //int* pBottom_label = m_labels->mutable_cpu_data();
}

/*
 * Data layer constructor
 */
template<typename Ntype>
DataLayer<Ntype>::DataLayer(string name, const int rows, const int cols)
{
    this->m_name = name;
    this->m_inputName = " ";
    this->m_bottom = NULL;
    this->m_top = NULL;
    this->m_loss = 0;
    this->m_lrate = 0.0f;
    m_labels = NULL;
    this->m_prevLayer.clear();
    this->m_nextLayer.clear();
    m_data_transformer = NULL;

    m_batchSize = ConfigTable::getInstance()->getBatchSize();
    this->m_inputChannels = ConfigTable::getInstance()->getChannels();
    DataLayerConfig* first_layer =(DataLayerConfig*)ConfigTable::getInstance()->getFirstLayer();
    // preprocessing param
    m_isDataTransFormer = first_layer->getDataTransformer();
    if(m_isDataTransFormer)
    {
        m_cropSize = first_layer->getCropSize();
        m_doMirror = first_layer->getDoMirror();
        m_scale = first_layer->getScale();
        // create object
        m_data_transformer = new DataTransformer<Ntype>(m_cropSize, m_doMirror, m_scale);
    }

    this->m_number = m_batchSize;
    this->m_channels = this->m_inputChannels;
    this->m_height = rows;
    this->m_width = cols;
    // reshape the top and label
    ReShape();
}

/*
 * Data Layer Forward propagation
 */
template<typename Ntype>
void DataLayer<Ntype>::Forward(Phase phase)
{
    //nothing
}

/*
 * Get batch size image
 */
template<typename Ntype>
void DataLayer<Ntype>::load_batch(int index, NDMatrix<Ntype>& image_data, NDMatrix<int>& image_label)
{
    m_dataSize = image_data.ND_num();
    int start = index * m_batchSize;

    Ntype* pBottom = this->m_bottom->mutable_cpu_data();
    const Ntype* pImg = image_data.cpu_data();
    int* pBottom_label = m_labels->mutable_cpu_data();
    const int* pImg_label = image_label.cpu_data();

    int k = 0;
    for (int i = start; i < ((start + m_batchSize) > m_dataSize ? m_dataSize : (start + m_batchSize)); i++) {
        for (int c = 0; c < this->m_channels; c++) {
            for (int h = 0; h < this->m_height; h++) {
                for (int w = 0; w < this->m_width; w++) {
                    int offset = image_data.ND_offset(i, c, h, w);
                    pBottom[this->m_bottom->ND_offset(k, c, h, w)] = pImg[offset];
                }
            }
        }
        pBottom_label[k] = pImg_label[i];
        k++;
    }
    //preprocessing
    //if(m_isDataTransFormer)
    //    m_data_transformer->Transform(this->m_bottom, this->m_top, TEST);
}

/*
 * Get batchSize image in random format
 */
template<typename Ntype>
void DataLayer<Ntype>::random_load_batch(NDMatrix<Ntype>& image_data, NDMatrix<int>& image_label)
{
    srand((unsigned)time(NULL));
    m_dataSize = image_data.ND_num();
    
     Ntype* pBottom = this->m_bottom->mutable_cpu_data();
    const Ntype* pImg = image_data.cpu_data();
    int* pBottom_label = m_labels->mutable_cpu_data();
    const int* pImg_label = image_label.cpu_data();

    int randomNum = ((long)rand() + (long)rand()) % (m_dataSize - m_batchSize);

    for(int i = 0; i< m_batchSize; i++)
    {
        for(int c = 0; c < this->m_channels; c++)
        {
            for(int h = 0; h < this->m_height; h++)
            {
                for(int w = 0; w < this->m_width; w++)
                {
                    int offset = image_data.ND_offset(i + randomNum, c, h, w);
                    pBottom[this->m_bottom->ND_offset(i, c, h, w)] = pImg[offset];
                }
            }
        }
        pBottom_label[i] = pImg_label[i + randomNum];
    }
    //preprocessing
    if(m_isDataTransFormer)
        m_data_transformer->Transform(this->m_bottom, this->m_top, TRAIN);
}

/*
 * Data Layer Backward propagation
 */
template<typename Ntype>
void DataLayer<Ntype>::Backward()
{
    //nothing
}


INSTANTIATE_CLASS(DataLayer);
