/*************************************************************************
	> File Name: nDMatrix.cpp
	> Author: TDX 
	> Mail: SA614149@mail.ustc.edu.cn
	> Created Time: 2016年12月09日 星期五 14时47分06秒
 ************************************************************************/

#include"nDMatrix.hpp"
#include"common.hpp"
#include"math_function.hpp"
#include<iostream>
#include<sstream>
#include<limits.h>
#include<glog/logging.h>

using namespace std;

/*
* for 4-D Matrix
*/
template<typename Ntype>
NDMatrix<Ntype>::NDMatrix(const int number, const int channels, const int height, const int width)
:m_capacity(0), m_data(), m_diff()
{
    ND_reShape(number, channels, height, width);
}

/*
* for N-D Matrix
*/
template<typename Ntype>
NDMatrix<Ntype>::NDMatrix(const vector<int>& shape):
m_capacity(0), m_data(), m_diff()
{
    ND_reShape(shape);
}

/*
* Display the dim
*/
template<typename Ntype>
string NDMatrix<Ntype>::ND_shape_string()
{
    ostringstream stream;
    for(int i = 0; i < m_ND_shape.size(); i++)
    {
        stream << m_ND_shape[i] <<" ";
    }

    stream << "(" << m_count << ")";
    return stream.str();
}

template<typename Ntype>
int NDMatrix<Ntype>::ND_count(int start, int end) 
{
    CHECK_LE(start, end);
    CHECK_GE(start, 0);
    CHECK_GE(end, 0);
    CHECK_LT(start, ND_size());
    CHECK_LE(end, ND_size());
    
    int count = 1;
    for(int i = start; i < end; i++)
    {
        count *= m_ND_shape[i];
    }

    return count;
}

template<typename Ntype>
int NDMatrix<Ntype>::ND_count(int start)
{
    return ND_count(start, ND_size());
}

/*
* for 4-D Matrix
*/
template<typename Ntype>
int NDMatrix<Ntype>::ND_offset(const int n, const int c, const int h, const int w) const
{
    CHECK_GE(n, 0);
    CHECK_LT(n, ND_num());
    CHECK_GE(c, 0);
    CHECK_LT(c, ND_channels());
    CHECK_GE(h, 0);
    CHECK_LT(h, ND_height());
    CHECK_GE(w, 0);
    CHECK_LT(w, ND_width());

    return ((n * ND_channels() + c) * ND_height() + h) * ND_width() + w;
}

/*
* 4-D NDMatrix
*/
template<typename Ntype>
void NDMatrix<Ntype>::ND_reShape(const int num, const int channels, const int height, const int width)
{
    vector<int> shape(4);
    shape[0] = num;
    shape[1] = channels;
    shape[2] = height;
    shape[3] = width;

    ND_reShape(shape);   
}

//for N-D Matrix
template<typename Ntype>
void NDMatrix<Ntype>::ND_reShape(vector<int> shape)
{
    if(4 == shape.size())
    {
        m_number = shape[0];
        m_channels = shape[1];
        m_height = shape[2];
        m_width = shape[3];
    }

    m_ND_shape.resize(shape.size());
    if(0 == shape.size())
    {
        m_count = 0; 
    }else
        m_count = 1;

    for(int i = 0; i < shape.size(); i++)
    {
        CHECK_GE(shape[i], 0);
        if(m_count != 0)
        {
            CHECK_LE(shape[i], INT_MAX / m_count) << "NDMatrix size exceeds INT_MAX";
        }
        m_count *= shape[i];
        m_ND_shape[i] = shape[i];
    } 

    if(m_count > m_capacity)
    {
        m_capacity = m_count;
        m_data.reset(new SyncedMemory(m_capacity * sizeof(Ntype)));
        m_diff.reset(new SyncedMemory(m_capacity * sizeof(Ntype)));
    }
}

template<typename Ntype>
void NDMatrix<Ntype>::CopyFrom(const NDMatrix<Ntype>& source)
{
    
}

// read only data 
template<typename Ntype>
const Ntype* NDMatrix<Ntype>::cpu_data()const
{
    CHECK(m_data);
    return (const Ntype*)m_data->cpu_data();
}

template<typename Ntype>
void NDMatrix<Ntype>::set_cpu_data(Ntype* data)
{
   CHECK(data);
   m_data->set_cpu_data(data);
}

template<typename Ntype>
const Ntype* NDMatrix<Ntype>::gpu_data() const
{
    CHECK(m_data);
    return (const Ntype*) m_data->gpu_data();
}

// read only diff
template<typename Ntype>
const Ntype* NDMatrix<Ntype>::cpu_diff() const
{
    CHECK(m_diff);
    return (const Ntype*) m_diff->cpu_data();
}

template<typename Ntype>
const Ntype* NDMatrix<Ntype>::gpu_diff() const
{
    CHECK(m_diff);
    return (const Ntype*) m_diff->gpu_data();
}

// read write data
template<typename Ntype>
Ntype* NDMatrix<Ntype>::mutable_cpu_data()
{
    CHECK(m_data);
    return static_cast<Ntype*>(m_data->mutable_cpu_data());
}

template<typename Ntype>
Ntype* NDMatrix<Ntype>::mutable_gpu_data()
{
    CHECK(m_data);
    return static_cast<Ntype*>(m_data->mutable_gpu_data());
}

//read write diff
template<typename Ntype>
Ntype* NDMatrix<Ntype>::mutable_cpu_diff()
{
    CHECK(m_diff);
    return static_cast<Ntype*>(m_diff->mutable_cpu_data());
}

template<typename Ntype>
Ntype* NDMatrix<Ntype>::mutable_gpu_diff()
{
    CHECK(m_diff);
    return static_cast<Ntype*>(m_diff->mutable_gpu_data());
}

template<typename Ntype>
void NDMatrix<Ntype>::update()
{
   switch(m_data->head())
    {
        case SyncedMemory::HEAD_AT_CPU:
        //perform computation on GPU
        cpu_axpy<Ntype>(m_count, Ntype(-1), static_cast<const Ntype*>(m_diff->cpu_data()), static_cast<Ntype*>(m_data->mutable_cpu_data()));
        break;

        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
        //perform computation on GPU



        break;
        default:
        LOG(FATAL) << "Syncedmem not initialized." ;
    } 
}

template<typename Ntype>
void NDMatrix<Ntype>::scale_data(Ntype scale_factor)
{
    Ntype* data;
    if(!m_data){ return; }
    switch(m_data->head())
    {
        case SyncedMemory::HEAD_AT_CPU:
        data = mutable_cpu_data();
        //perform computation on CPU
        cpu_scal<Ntype>(m_count, scale_factor, data);
        return;
        

        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
        data = mutable_gpu_data();
        //perform computation on GPU


        return;

        case SyncedMemory::UNINITIALIZED:
        return;

        default:
        LOG(FATAL) << "Unknown SyncedMemory head state: " << m_data->head();
    }
}

template<typename Ntype>
void NDMatrix<Ntype>::scale_diff(Ntype scale_factor)
{
    Ntype* diff;
    if(!m_diff){ return; }
    switch(m_diff->head())
    {
        case SyncedMemory::HEAD_AT_CPU:
        diff = mutable_cpu_diff();
        cpu_scal<Ntype>(m_count, scale_factor, diff);
        return;

        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
        //perform computation on GPU
        

        return;
        
        case SyncedMemory::UNINITIALIZED:
        return;

        default:
        LOG(FATAL) << "Unknown SyncedMemory head state: " << m_diff->head();
    }
}

// Instantiate class NDMatrix
INSTANTIATE_CLASS(NDMatrix);
