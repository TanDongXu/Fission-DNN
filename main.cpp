/*
* main.cpp
*
*  Created on: Nov 19, 2015
*      Author: tdx
*/
#include<iostream>
#include<glog/logging.h>
#include"common/nDMatrix.hpp"

int main(int argc, char** argv)
{
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging(argv[0]);

    //first test
    cout<<"First test: "<<endl;
    NDMatrix<float> a;
    cout<<"size: "<< a.ND_shape_string()<<endl;
    a.ND_reShape(1,2,3,4);
    cout<<"Size: "<< a.ND_shape_string()<<" "<<a.ND_shape()[0]<<endl;
    cout<<"size: "<<a.ND_size()<<" count: "<<a.ND_count()<<endl;
    
    for(int n = 0; n < a.ND_num();n++)
    {
        for(int c = 0; c < a.ND_channels(); c++)
        {
            for(int h = 0; h < a.ND_height();h++)
            {
                for(int w = 0; w < a.ND_width(); w++)
                {
                    cout<<a.ND_offset(n,c,h,w)<<" ";
                }
            }
        }
    }

    cout<<endl;
    cout<<a.ND_count(0,2)<<endl;
    cout<<a.ND_count(2)<<endl;;

    cout<<a.ND_num()<<a.ND_channels()<<a.ND_height()<<a.ND_width()<<endl;
    vector<int>b;
    a.ND_reShape(b);
    cout<<a.ND_shape_string()<<endl;
    //second test
    cout<<"Second test: "<<endl;
     NDMatrix<float>d;
    cout<<" b Size: "<<d.ND_shape_string()<<endl;
    d.ND_reShape(1,2,3,4);
    cout<<"b Size: "<< d.ND_shape_string()<<endl;
    float *p = d.mutable_cpu_data();
    
    for(int i = 0; i < d.ND_count(); i++)
    {
        p[i] = i;
    }

    for(int n = 0; n < d.ND_num(); n++)
    {
        for(int c = 0; c < d.ND_channels(); c++)
        {
            for(int h = 0; h < d.ND_height(); h++)
            {
                for(int w = 0; w < d.ND_width(); w++)
                {
                    cout<<"a["<<n<<"]["<<c<<"]["<<h<<"]["<<w<<"] = "<<d.data_at(n,c,h,w)<<endl;
                }
            }
        }
    }
    cout<<"data and diff merge: "<<endl;
    float* q = d.mutable_cpu_diff();
    for(int i = 0; i < d.ND_count(); i++)
    {
        q[i] = d.ND_count() - 1 - i;
    }

    d.update();
    for(int n = 0; n < d.ND_num(); n++)
    {
        for(int c = 0; c < d.ND_channels(); c++)
        {
            for(int h = 0; h < d.ND_height(); h++)
            {
                for(int w = 0; w < d.ND_width(); w++)
                {
                    cout<<"d["<<n<<"]["<<c<<"]["<<h<<"]["<<w<<"] = "<<d.data_at(n,c,h,w)<<endl;
                }
            }
        }
    }

    LOG(INFO)<< "Hello world";
    return 0;
}


