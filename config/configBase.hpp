/*************************************************************************
	> File Name: configBase.hpp
	> Author: TDX 
	> Mail: SA614149@mail.ustc.edu.cn
	> Created Time: 2016年12月13日 星期二 10时34分47秒
 ************************************************************************/

#ifndef _CONFIGBASE_HPP_
#define _CONFIGBASE_HPP_

#include<string>
#include<vector>
#include<map>
#include<glog/logging.h>
#include"common/util/util.cuh"

using namespace std;

class ConfigNonLinearity;
class ConfigPoolMethod;

class BaseLayerConfig
{
    public:
    string getType() const { return m_type; }
    string getName() const { return m_name; }
    string getInput() const { return m_input; }
    string getSubInput() const { return m_subInput; }
    const vector<BaseLayerConfig*> getVecNext() const { return m_next; }
    const vector<BaseLayerConfig*> getVecPrev() const { return m_prev; }
    vector<BaseLayerConfig*>& mutable_vecNext() { return m_next; }
    vector<BaseLayerConfig*>& mutable_vecPrev() { return m_prev; }
    
    protected:
    string m_name;
    string m_type;
    string m_input;
    string m_subInput;
    vector<BaseLayerConfig*> m_next;
    vector<BaseLayerConfig*> m_prev;
};

class ConfigTable
{
    public:
    static ConfigTable* getInstance()
    {
        static ConfigTable* config = new ConfigTable();
        return config;
    }

    void initConfig(string config_filename);
    inline int getBatchSize() const { return m_batchSize; }
    inline int getTrainEpochs() const { return m_trainEpochs; }
    inline int getIter_perEpoch() const { return m_iter_perEpoch; }
    inline int getChannels() const { return m_channels; }
    inline string getSolver_mode() const { return m_solver_mode; }
    inline BaseLayerConfig* getFirstLayer() const { return m_firstLayer; }
    inline BaseLayerConfig* getLastLayer() const { return m_lastLayer; }
    inline int getLayersNum() const { return m_mapLayers.size(); }
    void layersInsertByName(string name, BaseLayerConfig* layer);
    BaseLayerConfig* getLayerByName(string name);

    private:
    ConfigTable():m_nonLinearConfig(NULL), m_poolConfig(NULL), m_firstLayer(NULL), m_lastLayer(NULL){}
    void deleteSpace();
    void deleteComments();
    string textToString(string config_filename);
    int getIntVariable(string& inputStr, string varName);
    float getFloatVariable(string& inputStr, string varName);
    string getStringVariable(string& inputStr, string strName);
    vector<string> getVectorVariable(string& inputStr, string strName);
    void showLayersConfig();
    int m_batchSize;
    int m_trainEpochs;
    int m_iter_perEpoch;
    int m_channels;
    string m_solver_mode;
    string m_textString;
    ConfigNonLinearity* m_nonLinearConfig;
    ConfigPoolMethod* m_poolConfig;
    BaseLayerConfig* m_firstLayer;
    BaseLayerConfig* m_lastLayer;
    map<string, BaseLayerConfig*> m_mapLayers;
};

class ConfigNonLinearity
{
    public:
    ConfigNonLinearity(string method)
    {
        if(method == string("NL_SIGMOID"))
        {
            m_nonLinearType = ACTIVATION_SIGMOID;
        }else if(method == string("NL_TANH"))
        {
            m_nonLinearType = ACTIVATION_TANH;
        }else if(method == string("NL_RELU"))
        {
            m_nonLinearType = ACTIVATION_RELU;
        }else if(method == string("NL_LRELU"))
        {
            m_nonLinearType = ACTIVATION_LRELU;
        }else
        {
            LOG(FATAL) << "Unknown NonLinearity type " << method;
        }
    }

    int getNonLinearType() const { return m_nonLinearType; }

    private:
    int m_nonLinearType;
};

class ConfigPoolMethod
{
    public:
    ConfigPoolMethod(string method)
    {
        if(method == string("POOL_MAX"))
        {
            m_poolType = POOL_MAX;
        }else if(method == string("POOL_AVE_INCLUDE_PAD"))
        {
            m_poolType = POOL_AVERAGE_COUNT_INCLUDE_PADDING;
        }else if(method == string("POOL_AVE_EXCLUDE_PAD"))
        {
            m_poolType = POOL_AVERAGE_COUNT_EXCLUDE_PADDING;
        }else
        {
            LOG(FATAL) << "Unknown poolType " << method;
        }
    }

    int getPoolType() const { return m_poolType; }

    private:
    int  m_poolType;
};

class DataLayerConfig : public BaseLayerConfig
{
    public:
    DataLayerConfig(string type, string name, string input, string subInput)
    {
        m_type = type;
        m_name = name;
        m_input = input;
        m_subInput = subInput;
    }
};

class ConvLayerConfig : public BaseLayerConfig
{
    public:
    ConvLayerConfig(string type, string name, string input, string subInput,
                    int kernelSize, int pad_h, int pad_w, int stride_h, int stride_w,
                     int kernelAmount, float init_w, float lrate, float weight_decay)
    {
        m_type = type;
        m_name = name;
        m_input = input;
        m_subInput = subInput;
        m_kernelSize = kernelSize;
        m_pad_h = pad_h;
        m_pad_w = pad_w;
        m_stride_h = stride_h;
        m_stride_w = stride_w;
        m_kernelAmount = kernelAmount;
        m_init_w = init_w;
        m_lrate = lrate;
        m_weight_decay = weight_decay;
    }

    int getKernelSize() const { return m_kernelSize; }
    int getPad_h() const { return m_pad_h; }
    int getPad_w() const { return m_pad_w; }
    int getStride_h() const { return m_stride_h; }
    int getStride_w() const { return m_stride_w; }
    int getKernelAmount() const { return m_kernelAmount; }
    float getInit_w() const { return m_init_w; }
    float getLrate() const { return m_lrate; }
    float getWeight_decay() const { return m_weight_decay; }

    private:
    int m_kernelSize;
    int m_pad_h;
    int m_pad_w;
    int m_stride_h;
    int m_stride_w;
    int m_kernelAmount;
    float m_init_w;
    float m_lrate;
    float m_weight_decay;
};

class PoolLayerConfig : public BaseLayerConfig
{
    public:
    PoolLayerConfig(string type, string name, string input, string subInput, int size,
                    int pad_h, int pad_w, int stride_h,int stride_w, int poolType)
    {
        m_type = type;
        m_name = name;
        m_input = input;
        m_subInput = subInput;
        m_size = size;
        m_pad_h = pad_h;
        m_pad_w = pad_w;
        m_stride_h = stride_h;
        m_stride_w = stride_w;
        m_poolType = poolType;
    }

    int getSize() const { return m_size; }
    int getPad_h() const { return m_pad_h; }
    int getPad_w() const { return m_pad_w; }
    int getStride_h() const { return m_stride_h; }
    int getStride_w() const { return m_stride_w; }
    int getPoolType() const { return m_poolType; }

    private:
    int m_size;
    int m_pad_h;
    int m_pad_w;
    int m_stride_h;
    int m_stride_w;
    int m_poolType;
};

class InceptionLayerConfig : public BaseLayerConfig
{
    public:
    InceptionLayerConfig(string type, string name, string input, string subInput,
                        int one, int three, int five, int three_reduce, int five_reduce,
                        int pool_proj, float init_w, float lrate, float weight_decay)
    {
        m_type = type;
        m_name = name;
        m_input = input;
        m_subInput = subInput;
        m_one = one;
        m_three = three;
        m_five = five;
        m_three_reduce = three_reduce;
        m_five_reduce = five_reduce;
        m_pool_proj = pool_proj;
        m_init_w = init_w;
        m_lrate = lrate;
        m_weight_decay = weight_decay;
    }

    int getOne() const { return m_one; }
    int getThree() const { return m_three; }
    int getFive() const { return m_five; }
    int getThree_reduce() const { return m_three_reduce; }
    int getFive_reduce() const { return m_five_reduce; }
    int getPool_proj() const { return m_pool_proj; }
    float getInit_w() const { return m_init_w; }
    float getLrate() const { return m_lrate; }
    float getWeight_decay() const { return m_weight_decay; }

    private:
    int m_one;
    int m_three;
    int m_five;
    int m_three_reduce;
    int m_five_reduce;
    int m_pool_proj;
    float m_init_w;
    float m_weight_decay;
    float m_lrate;
};

class BranchLayerConfig : public BaseLayerConfig
{
    public:
    BranchLayerConfig(string type, string name, string input, string subInput, vector<string>outputs )
    {
        m_type = type;
        m_name = name;
        m_input = input;
        m_subInput = subInput;
        m_outputs = outputs;
    }

    vector<string> getOutputs() const { return m_outputs; }

    private:
    vector<string> m_outputs;
};

class HiddenLayerConfig : public BaseLayerConfig
{
    public:
    HiddenLayerConfig(string type, string name, string input, string subInput,
                      int NumHiddenNeurons, float init_w, float lrate, float weight_decay)
    {
        m_type = type;
        m_name = name;
        m_input = input;
        m_subInput = subInput;
        m_numHiddenNeurons = NumHiddenNeurons;
        m_init_w = init_w;
        m_lrate = lrate;
        m_weight_decay = weight_decay;
    }

    int getNumNeurons() const { return m_numHiddenNeurons; }
    float getInit_w() const { return m_init_w; }
    float getLrate() const {  return m_lrate; }
    float getWeight_decay() const { return m_weight_decay; }

    private:
    int m_numHiddenNeurons;
    float m_init_w;
    float m_lrate;
    float m_weight_decay;
};

class DropOutLayerConfig : public BaseLayerConfig 
{
    public:
    DropOutLayerConfig(string type, string name, string input, string subInput, float rate)
    {
        m_type = type;
        m_name = name;
        m_input = input;
        m_subInput = subInput;
        m_dropOut_rate = rate;
    }

    float getDropOut_rate() const { return m_dropOut_rate; }

    private:
    float m_dropOut_rate;
};

class ActivationLayerConfig : public BaseLayerConfig
{
    public:
    ActivationLayerConfig(string type, string name, string input, string subInput, int non_linearity)
    {
        m_type = type;
        m_name = name;
        m_input = input;
        m_subInput = subInput;
        m_nonLinearType = non_linearity;
    }

    int getNonLinearType() const { return m_nonLinearType; }

    private:
    int m_nonLinearType;
};

class LRNLayerConfig : public BaseLayerConfig
{
    public:
    LRNLayerConfig(string type, string name, string input, string subInput, unsigned lrnN, float lrnAlpha, float lrnBeta)
    {
        m_type = type;
        m_name = name;
        m_input = input;
        m_subInput = subInput;
        m_lrnN = lrnN;
        m_lrnAlpha = lrnAlpha;
        m_lrnBeta = lrnBeta;
    }

    unsigned getLrnN() const { return m_lrnN; }
    float getLrnAlpha() const { return m_lrnAlpha; }
    float getLrnBeta() const { return m_lrnBeta; }

    private:
    unsigned m_lrnN;
    float m_lrnAlpha;
    float m_lrnBeta;
};

class SoftMaxLayerConfig : public BaseLayerConfig
{
    public:
    SoftMaxLayerConfig(string type, string name, string input, string subInput,
                       int nClasses, float weight_decay)
    {
        m_type = type;
        m_name = name;
        m_input = input;
        m_subInput = subInput;
        m_nClasses = nClasses;
        m_weight_decay = weight_decay;
    }

    int getNClasses() const { return m_nClasses; }
    float getWeight_decay() const{ return m_weight_decay; }

    private:
    int m_nClasses;
    float m_weight_decay;
};

#endif
