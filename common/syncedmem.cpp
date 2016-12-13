/*************************************************************************
	> File Name: syncedmem.cpp
	> Author: TDX 
	> Mail: SA614149@mail.ustc.edu.cn
	> Created Time: 2016年12月11日 星期日 14时15分51秒
 ************************************************************************/
#include"syncedmem.hpp"
#include<iostream>
using namespace std;

SyncedMemory::~SyncedMemory()
{
    //free cpu and GPU memory
    freeHostMem(m_cpu_ptr, m_cpuMalloc_use_cuda);
    freeDeviceMem(m_gpu_ptr);
}

const void* SyncedMemory::cpu_data()
{
    to_cpu();
    return (const void*)m_cpu_ptr;
}

void SyncedMemory::set_cpu_data(void* data)
{
    CHECK(data);
    if(NULL != m_cpu_ptr)
    {
        freeHostMem(m_cpu_ptr, m_cpuMalloc_use_cuda);
    }

    m_cpu_ptr = data;
    m_head = HEAD_AT_CPU;
}

const void* SyncedMemory::gpu_data()
{
    to_gpu();
    return (const void*)m_gpu_ptr;
}

void SyncedMemory::set_gpu_data(void* data)
{
    CHECK(data);
    if(NULL != m_gpu_ptr)
    {
        freeDeviceMem(m_gpu_ptr);
    }
    m_gpu_ptr = data;
    m_head = HEAD_AT_GPU;
}

void* SyncedMemory::mutable_cpu_data()
{
    to_cpu();
    m_head = HEAD_AT_CPU;
    return m_cpu_ptr;
}

void* SyncedMemory::mutable_gpu_data()
{
    to_gpu();
    m_head = HEAD_AT_GPU;
    return m_gpu_ptr;
}

void SyncedMemory::to_cpu()
{
    switch(m_head)
    {
        case UNINITIALIZED:
            mallocHostMem(&m_cpu_ptr, m_size, m_cpuMalloc_use_cuda);
            cpuMemoryMemset(m_cpu_ptr, m_size);
            m_head = HEAD_AT_CPU;
            break;
        case HEAD_AT_GPU:
            if(NULL == m_cpu_ptr)
            {
                mallocHostMem(&m_cpu_ptr, m_size, m_cpuMalloc_use_cuda);
            }
            mem_gpu2cpu(m_cpu_ptr, m_gpu_ptr, m_size);
            m_head = SYNCED;
            break;
        case HEAD_AT_CPU:
        case SYNCED:
        break;

        default:
        LOG(FATAL) << "Unknown SyncedMemory heads state: " << m_head;
    }
}


void SyncedMemory::to_gpu()
{
    switch(m_head)
    {
        case UNINITIALIZED:
            mallocDeviceMem(&m_gpu_ptr, m_size);
            gpuMemoryMemset(m_gpu_ptr, m_size);
            m_head = HEAD_AT_GPU;
            break;

        case HEAD_AT_CPU:
            if(NULL == m_gpu_ptr)
            {
                mallocDeviceMem(&m_gpu_ptr,m_size);
            }
            mem_cpu2gpu(m_gpu_ptr, m_cpu_ptr, m_size);
            m_head = SYNCED;
            break;

        case HEAD_AT_GPU:
        case SYNCED:
        break;

        default:
        LOG(FATAL) << "Unknown SyncedMemory heads state: " << m_head;
    }
}

void SyncedMemory::async_gpu_push(const cudaStream_t& stream)
{
    CHECK(m_head == HEAD_AT_CPU);
    CHECK(m_cpuMalloc_use_cuda);
    if(NULL == m_gpu_ptr)
    {
        mallocDeviceMem(&m_gpu_ptr, m_size);
    }

    const cudaMemcpyKind put = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpyAsync(m_gpu_ptr, m_cpu_ptr, m_size, put, stream));
    m_head = SYNCED;
}

