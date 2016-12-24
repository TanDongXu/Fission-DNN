/*************************************************************************
	> File Name: layer.cpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月17日 星期六 09时54分26秒
 ************************************************************************/

#include<iostream>
#include"layer.hpp"
#include<glog/logging.h>
#include"config/configBase.hpp"

using namespace std;

template<typename Ntype>
Layer<Ntype>* LayerContainer<Ntype>::getLayerByName(string name)
{
    if(m_layersMap.find(name) != m_layersMap.end())
    {
        return m_layersMap[name];
    }else
    {
        LOG(FATAL) << "Layer " << name << " is not in the layersContainer.";
        return NULL;
    }
}

//  Story the layers into map
template<typename Ntype>
void LayerContainer<Ntype>::storelayer(string prevlayerName, string name, Layer<Ntype>* layer)
{
    if(m_layersMap.find(name) == m_layersMap.end())
    {
        m_layersMap[name] = layer;
        storeName(name);

        // Create a linked list
        if(string("NULL") == prevlayerName)
        {
            m_layersMap[name]->getPrevLayerConfig().clear();
            m_layersMap[name]->setInputName(" ");
        }else
        {
            m_layersMap[name]->setInputName(prevlayerName);
            //cout<<"prevName: "<<prev_name<<" name: "<<name<<endl;
            m_layersMap[prevlayerName]->insertNextLayer(m_layersMap[name] );
            m_layersMap[name]->insertPrevLayer(m_layersMap[prevlayerName]);
        }

    }else
    {
        LOG(FATAL) << name <<" has already in layersMap.";
    }
}

/*
 * overload
 * Linear storage layer
 */
template<typename Ntype>
void LayerContainer<Ntype>::storelayer(string name, Layer<Ntype>* layer)
{
    if(m_layersMap.find(name) == m_layersMap.end())
    {
        m_layersMap[name] = layer;
        storeName(name);

        // Create a linked list
        if(1 == m_layersMap.size())
        {
            m_layersMap[name]->getPrevLayerConfig().clear();
            m_layersMap[name]->setInputName(" ");

        }else
        {
            m_layersMap[name]->setInputName(m_layersMap[m_layersName[m_layersName.size() - 2]]->getName());
            m_layersMap[m_layersName[m_layersName.size() -2 ]]->insertNextLayer(m_layersMap[name]);
            m_layersMap[name]->insertPrevLayer(m_layersMap[m_layersName[m_layersName.size() - 2]]);
        }
    }else
    {
        LOG(FATAL) << name <<" has already in layersMap.";
    }
}


INSTANTIATE_CLASS(Layer);
INSTANTIATE_CLASS(LayerContainer);
