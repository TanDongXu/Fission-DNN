/*************************************************************************
    > File Name: nDMatrix.hpp
    > Author: TDX 
    > Mail: SA614149@mail.ustc.edu.cn
    > Created Time: 2016年12月09日 星期五 14时25分21秒
 ************************************************************************/
#ifndef NDMATRIX_HPP_
#define NDMATRIX_HPP_

#include<iostream>
#include<vector>
#include<string>
#include<boost/shared_ptr.hpp>

#include"syncedmem.hpp"

using namespace std;

template<typename Ntype>
class NDMatrix
{
    public:
    NDMatrix():m_number(0), m_channels(0), m_height(0), m_width(0), m_capacity(0), m_count(0), m_data(), m_diff(){}
    //for N-D Matrix
    NDMatrix(const vector<int>& shape);
    //for 4-D Matrix
    NDMatrix(const int number, const int channels, const int height, const int width);
    void ND_reShape(const int num, const int channels, const int height, const int width);
    //for N-D Matrix
    void ND_reShape(vector<int> shape);
    string ND_shape_string();
    inline const vector<int>& ND_shape() const { return m_ND_shape; }
    inline int ND_size() const { return m_ND_shape.size(); }
    inline int ND_count() const { return m_count; }
    int ND_count(int start, int end);
    int ND_count(int start);
    //for 4-D Matrix
    inline int ND_num() const { return m_number; }
    inline int ND_channels() const { return m_channels; }
    inline int ND_height() const { return m_height; }
    inline int ND_width() const { return m_width; }
    int ND_offset(const int n, const int c = 0, const int h = 0, const int w = 0) const;
    //data and diff
    void CopyFrom(const NDMatrix<Ntype>& source);
    inline Ntype data_at(const int n, const int c, const int h, const int w) const { return cpu_data()[ND_offset(n, c, h, w)]; }
    inline shared_ptr<SyncedMemory>& data(){ CHECK(m_data); return m_data; }
    inline Ntype diff_at(const int n, const int c, const int h, const int w) const { return cpu_diff()[ND_offset(n, c, h, w)]; }
    inline shared_ptr<SyncedMemory>& diff(){ CHECK(m_diff); return m_diff; }

    // read only data and diff
    const Ntype* cpu_data() const;
    void set_cpu_data(Ntype* data);
    const Ntype* gpu_data() const;
    const Ntype* cpu_diff() const;
    const Ntype* gpu_diff() const;
    //read write data and diff
    Ntype* mutable_cpu_data();
    Ntype* mutable_gpu_data();
    Ntype* mutable_cpu_diff();
    Ntype* mutable_gpu_diff();

    //data and diff merge
    void update();
    // scale the NDMatrix data and diff by a constant factor
    void scale_data(Ntype scale_factor);
    void scale_diff(Ntype scale_factor);

    private:
    //for 4-D
    int m_number;
    int m_channels;
    int m_height;
    int m_width;
    //m_count is newest capacity
    int m_capacity;
    int m_count;
    shared_ptr<SyncedMemory> m_data;
    shared_ptr<SyncedMemory> m_diff;
    vector<int> m_ND_shape;

};


#endif   //_NDMATRIX_H_
