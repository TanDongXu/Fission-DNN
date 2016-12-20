/*************************************************************************
	> File Name: syncedmem.hpp
	> Author: TDX 
	> Mail: SA614149@mail.ustc.edu.cn
	> Created Time: 2016年12月11日 星期日 09时47分14秒
 ************************************************************************/

#ifndef _SYNCEDMEM_HPP_
#define _SYNCEDMEM_HPP_

#include"common.hpp"
#include<glog/logging.h>
#include<cuda.h>
#include<stdlib.h>
#include<cuda_runtime.h>

inline void mallocHostMem(void**ptr, size_t size, bool use_cuda)
{
    if(use_cuda)
    {
        if(NULL != (*ptr)) CUDA_CHECK(cudaFreeHost(ptr));
        CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
        return;
    }

    *ptr = malloc(size);
    CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void freeHostMem(void* ptr, bool use_cuda)
{
    if(NULL == ptr) return;
    if(use_cuda)
    {
        CUDA_CHECK(cudaFreeHost(ptr));
        return;
    }

    free(ptr);
}

inline void mallocDeviceMem(void** dev_ptr, size_t size)
{
    if(NULL != (*dev_ptr))
    {
        CUDA_CHECK(cudaFree(*dev_ptr));
    }
    CUDA_CHECK(cudaMalloc(dev_ptr, size));
}

inline void freeDeviceMem(void* dev_ptr)
{
    if(NULL == dev_ptr) return;
    CUDA_CHECK(cudaFree(dev_ptr));
}

inline void cpuMemoryMemset(void* host_ptr, size_t size)
{
    CHECK(host_ptr);
    CHECK_GT(size, 0);
    memset(host_ptr, 0, size);
}

inline void gpuMemoryMemset(void* dev_ptr, size_t size)
{
    CHECK(dev_ptr);
    CHECK_GT(size, 0);
    CUDA_CHECK(cudaMemset(dev_ptr, 0, size));
}

inline void mem_cpu2gpu(void* dev_ptr, void* host_ptr, size_t size)
{
    CHECK(host_ptr);
    CHECK(dev_ptr);
    CHECK_GT(size, 0);
    CUDA_CHECK(cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice));
}

inline void mem_gpu2cpu(void* host_ptr, void* dev_ptr, size_t size)
{
    CHECK(dev_ptr);
    CHECK(host_ptr);
    CHECK_GT(size, 0);
    CUDA_CHECK(cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost));
}

inline void mem_gpu2gpu(void* dst_ptr, void* src_ptr, size_t size)
{
    CHECK(src_ptr);
    CHECK(dst_ptr);
    CHECK_GT(size, 0);
    CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice));
}

inline void mem_cpu2cpu(void* dst_ptr, void* src_ptr, size_t size)
{
    CHECK(dst_ptr);
    CHECK(src_ptr);
    CHECK_GT(size, 0);
    memcpy(dst_ptr, src_ptr, size);
}

class SyncedMemory
{
    public:
    SyncedMemory():m_cpu_ptr(NULL), m_gpu_ptr(NULL), m_size(0), m_head(UNINITIALIZED), m_cpuMalloc_use_cuda(false){}
    SyncedMemory(size_t size):m_cpu_ptr(NULL), m_gpu_ptr(NULL), m_size(size),  m_head(UNINITIALIZED), m_cpuMalloc_use_cuda(false){}
    ~SyncedMemory();

    const void* cpu_data();
    void set_cpu_data(void* data);
    const void* gpu_data();
    void set_gpu_data(void* data);
    void* mutable_cpu_data();
    void* mutable_gpu_data();
    enum SyncedHead{UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED};
    SyncedHead head(){ return m_head; }
    size_t size(){ return m_size; }
    // if use pinned memory, must be set it first
    void setCpuMalloc_useCuda(){ m_cpuMalloc_use_cuda = true; }
    void async_gpu_push(const cudaStream_t& stream);
    
    private:
    void to_cpu();
    void to_gpu();
    void* m_cpu_ptr;
    void* m_gpu_ptr;
    size_t m_size;
    //if have write config, it must delete
    bool m_cpuMalloc_use_cuda;
    SyncedHead m_head;
};
#endif
