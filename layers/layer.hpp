/*************************************************************************
	> File Name: layer.hpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月16日 星期五 14时45分01秒
 ************************************************************************/

#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include<iostream>
#include<vector>
#include<map>
#include<string>
#include<glog/logging.h>
#include"common/common.hpp"
#include"common/nDMatrix.hpp"
#include"config/configBase.hpp"

using namespace std;

enum Phase{ TRAIN, TEST };

template<typename Ntype>
class Layer
{
    public:
    virtual ~Layer(){}
    // Adjust the shapes of top NDMatrix and internal buffers to accommodate the shapes of the bottom
    // Given the bottom NDMatrix, compute the top NDMatrix and the loss
    virtual void Forward(Phase phase) = 0;
    // given the top NDMatrix error gradients, compute the bottom NDMatrix error gradients
    virtual void Backward() = 0;
    inline void insertPrevLayer(Layer<Ntype>* layer){ CHECK(layer); m_prevLayer.push_back(layer); }
    inline void insertNextLayer(Layer<Ntype>* layer){ CHECK(layer); m_nextLayer.push_back(layer); }
    inline vector<Layer<Ntype>*>& getPrevLayerConfig() { return m_prevLayer; }
    inline vector<Layer<Ntype>*>& getNextLayerConfig() { return m_nextLayer; }
    inline void setInputName(string name){ m_inputName = name; }
    inline string getName() const { return m_name; }
    inline NDMatrix<Ntype>* getTop() const { return m_top; }

    protected: 
    virtual void ReShape() = 0;
    string m_name;
    string m_inputName;
    Ntype m_loss;
    float m_lrate;
    int m_inputChannels;
    int m_number;
    int m_channels;
    int m_height;
    int m_width;
    NDMatrix<Ntype>* m_bottom;
    NDMatrix<Ntype>* m_top;
    vector<Layer<Ntype>*> m_prevLayer;
    vector<Layer<Ntype>*> m_nextLayer;
};

template<typename Ntype>
class LayerContainer
{
    public:
    static LayerContainer<Ntype>* getInstance()
    {
        static LayerContainer<Ntype>* container = new LayerContainer<Ntype>();
        return container;
    }

    Layer<Ntype>* getLayerByName(string name);
    void storelayer(string name, Layer<Ntype>* layer);
    void storelayer(string prevlayerName, string name, Layer<Ntype>* layer);
    inline void storeName(string name) { m_layersName.push_back(name);}
    inline string getNameByIndex(int index)
    {
        CHECK_LT(index, m_layersName.size()) << "The index out of layers size.";
        return m_layersName[index];
    }
    inline int getLayersNum() const { return m_layersMap.size(); }
    inline bool isHasLayer(string name)
    {
        if(m_layersMap.find(name) != m_layersMap.end())
        {
            return true;
        }else
        {
            return false;
        }
    }
    
    private:
    LayerContainer(){}
    map<string, Layer<Ntype>*> m_layersMap;
    vector<string> m_layersName;
};
#endif
