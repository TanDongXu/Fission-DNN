/*************************************************************************
	> File Name: config_table.cpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月13日 星期二 12时34分43秒
 ************************************************************************/

#include<iostream>
#include<fstream>
#include<sstream>
#include<glog/logging.h>
#include"configBase.hpp"

using namespace std;

void ConfigTable::initConfig(string config_filename)
{
    m_textString = textToString(config_filename);
    deleteComments();
    deleteSpace();
    m_solver_mode = getStringVariable(m_textString, "SOLVER_MODE");
    m_batchSize = getIntVariable(m_textString, "BATCH_SIZE");
    m_channels = getIntVariable(m_textString, "CHANNELS");
    m_trainEpochs = getIntVariable(m_textString, "TRAIN_EPOCHS");
    m_iter_perEpoch = getIntVariable(m_textString, "ITER_PER_EPOCH");
    showLayersConfig();
}

// file text to string
string ConfigTable::textToString(string config_filename)
{
    ifstream config_file(config_filename, std::ios::in);
    CHECK(config_file) << "Unable to open file " << config_filename;
    ostringstream str_stream;
    str_stream << config_file.rdbuf();
    string str = str_stream.str();
    config_file.close();
    config_file.clear();
    
    return str;
}

// delete comments
void ConfigTable::deleteComments()
{
    size_t pos1, pos2;
    while(1)
    {
        pos1 = m_textString.find("/");
        if(pos1 == string::npos)  break;

        for(pos2 = pos1 + 1; pos2 < m_textString.size(); pos2++)
        {
            if('/' == m_textString[pos2])
            {
                break;
            }
        }

        m_textString.erase(pos1, pos2 - pos1 + 1);
    }
}

// delete space
void ConfigTable::deleteSpace()
{
    if(m_textString.empty()) return;
    size_t pos1, pos2, e, n, t;
    while(1)
    {
        e = m_textString.find(' ');
        t = m_textString.find('\t');
        n = m_textString.find('\n');

        if(e == string::npos && t == string::npos && n == string::npos)  break;
        if(e < t || t == string::npos) 
            pos1 = e;
        else
            pos1 = t;

        if(n <  pos1 || pos1 == string::npos) pos1 = n;
        for(pos2 = pos1 + 1; pos2 < m_textString.size(); pos2++)
        {
            if(!(m_textString[pos2] == '\t' || m_textString[pos2] == '\n' || m_textString[pos2] == ' '))
                break;
        }

        m_textString.erase(pos1, pos2 - pos1);

    }
}

int ConfigTable::getIntVariable(string& inputStr, string varName)
{
    size_t pos = inputStr.find(varName);
    CHECK(pos != string::npos) << varName << " Not Found in ConfigTable.";

    int i = pos + 1;
    int res = 1;
    while(1)
    {
        if(inputStr.length() == i) break;
        if(';' == inputStr[i]) break;
        ++i;
    }
    string sub = inputStr.substr(pos, i - pos + 1);
    if(';' == sub[sub.length() - 1])
    {
        string content = sub.substr(varName.length() + 1, sub.length() - varName.length() - 2);
        res = atoi(content.c_str());
    }

    inputStr.erase(pos, i - pos + 1);
    return res;
}

float ConfigTable::getFloatVariable(string& inputStr, string varName)
{
    size_t pos = inputStr.find(varName);
    CHECK(pos != string::npos) << varName << " Not Found in ConfigTable.";

    int i = pos + 1;
    float res = 0.0f;
    while(1)
    {
        if(inputStr.length() == i) break;
        if(';' == inputStr[i]) break;
        ++i;
    }
    string sub = inputStr.substr(pos, i - pos + 1);
    if(';' == sub[sub.length() - 1])
    {
        string content = sub.substr(varName.length() + 1, sub.length() - varName.length() - 2);
        res = atof(content.c_str());
    }

    inputStr.erase(pos, i - pos + 1);
    return res;
}

string ConfigTable::getStringVariable(string& inputStr, string strName)
{
    size_t pos = inputStr.find(strName);
    if(pos == string::npos) return "NULL";

    int i = pos + 1;
    while(1)
    {
        if(inputStr.length() == i) break;
        if(';' == inputStr[i]) break;
        ++i;
    }

    string sub = inputStr.substr(pos, i - pos + 1);
    string content;
    if(';' == sub[sub.length() - 1])
    {
        content = sub.substr(strName.length() + 1, sub.length() - strName.length() - 2);
    }
    inputStr.erase(pos, i - pos + 1);
    return content;
}

vector<string> ConfigTable::getVectorVariable(string& inputStr, string strName)
{
    vector<string> result;
    size_t pos = inputStr.find(strName);
    CHECK(pos != string::npos) << strName << " Not Found in ConfigTable.";

    int i = pos + 1;
    while(1)
    {
        if(i == inputStr.length()) break;
        if(inputStr[i] == ';') break;
        ++i;
    }

    string sub = inputStr.substr(pos, i - pos + 1);
    string content;
    if(';' == sub[sub.length() -1 ])
    {
        content = sub.substr(strName.length() + 1, sub.length() - strName.length() - 2);
    }

    inputStr.substr(pos, i - pos + 1);
    while(content.size())
    {
        size_t pos = content.find(',');
        if(pos == string::npos)
        {
            result.push_back(content);
            break;
        }else
        {
            result.push_back(content.substr(0, pos));
            content.erase(0, pos + 1);
        }
    }

    return result;
}

void ConfigTable::layersInsertByName(string name, BaseLayerConfig* layer)
{
    CHECK(layer);
    if(m_mapLayers.find(name) == m_mapLayers.end())
    {
        m_mapLayers[name] = layer;
    }else
    {
        LOG(FATAL) << name <<  " have already exist.";
    }
}

BaseLayerConfig* ConfigTable::getLayerByName(string name)
{
    if(m_mapLayers.find(name) != m_mapLayers.end())
    {
        return m_mapLayers[name];
    }else
    {
        LOG(FATAL) << name <<" layer does not exit."; 
    }
}

void ConfigTable::showLayersConfig()
{
    vector<string> vStrLayers;
    if(m_textString.empty()) return;
    int head = 0;
    int tail = 0;
    while(1)
    {
        if(head == m_textString.length()) break;
        if(m_textString[head] == '['){
            tail = head + 1;
            while(1)
            {
                if(tail == m_textString.length()) break;
                if(m_textString[tail] == ']') break;
                ++tail;
            }
            string sub = m_textString.substr(head, tail - head + 1);
            if(sub[sub.length() - 1] == ']')
            {
                //delete last ']'
                sub.erase(sub.begin() + sub.length() - 1);
                //delete first '['
                sub.erase(sub.begin());
                vStrLayers.push_back(sub);
            }

            m_textString.erase(head, tail - head + 1);
        }else 
            ++head;
    }

    cout << endl << endl ;
    LOG(INFO) << "Read The Layers Configure :";
    for(int i = 0; i < vStrLayers.size(); i++)
    {
        string type = getStringVariable(vStrLayers[i], "LAYER");
        string name = getStringVariable(vStrLayers[i], "NAME");
        string input = getStringVariable(vStrLayers[i], "INPUT");
        string sub_input = getStringVariable(vStrLayers[i], "SUB_INPUT");
        
        BaseLayerConfig* layer;
        if(string("CONV") == type)
        {
            int ks = getIntVariable(vStrLayers[i], "KERNEL_SIZE");
            int ka = getIntVariable(vStrLayers[i], "KERNEL_AMOUNT");
            int pad_h = getIntVariable(vStrLayers[i], "PAD_H");
            int pad_w = getIntVariable(vStrLayers[i], "PAD_W");
            int stride_h = getIntVariable(vStrLayers[i], "STRIDE_H");
            int stride_w = getIntVariable(vStrLayers[i], "STRIDE_W");

            float init_w = getFloatVariable(vStrLayers[i], "INIT_W");
            float lrate = getFloatVariable(vStrLayers[i], "LEARN_RATE");
            float weight_decay = getFloatVariable(vStrLayers[i], "WEIGHT_DECAY");

            layer = new ConvLayerConfig(type, name, input, sub_input, ks, pad_h, pad_w, stride_h,
                                   stride_w, ka, init_w, lrate, weight_decay);

            cout << endl;
            LOG(INFO) << "***********************Conv layer**********************";
            cout << endl;
            LOG(INFO) << "              NAME : " << name;
            LOG(INFO) << "             INPUT : " << input;
            LOG(INFO) << "         SUB_INPUT : " << sub_input;
            LOG(INFO) << "       KERNEL_SIZE : " << ks;
            LOG(INFO) << "     KERNEL_AMOUNT : " << ka;
            LOG(INFO) << "             PAD_H : " << pad_h;
            LOG(INFO) << "             PAD_W : " << pad_w;
            LOG(INFO) << "          STRIDE_H : " << stride_h;
            LOG(INFO) << "          STRIDE_W : " << stride_w;
            LOG(INFO) << "            INIT_W : " << init_w;
            LOG(INFO) << "        LEARN_RATE : " << lrate;
            LOG(INFO) << "      WEIGHT_DECAY : " << weight_decay;

        }else if(string("POOLING") == type)
        {
            string poolType = getStringVariable(vStrLayers[i], "POOLING_TYPE");
            m_poolConfig = new ConfigPoolMethod(poolType);
            int size = getIntVariable(vStrLayers[i], "POOLDIM");
            int pad_h = getIntVariable(vStrLayers[i], "PAD_H");
            int pad_w = getIntVariable(vStrLayers[i], "PAD_W");
            int stride_h = getIntVariable(vStrLayers[i], "STRIDE_H");
            int stride_w = getIntVariable(vStrLayers[i], "STRIDE_W");

            layer = new PoolLayerConfig(type, name, input, sub_input, size, pad_h, pad_w, stride_h,
                                      stride_w, m_poolConfig->getPoolType());

            cout << endl;
            LOG(INFO) << "***********************Pooling layer*******************";
            cout << endl;
            LOG(INFO) << "              NAME : " << name;
            LOG(INFO) << "             INPUT : " << input;
            LOG(INFO) << "         SUB_INPUT : " << sub_input;
            LOG(INFO) << "      POOLING_TYPE : " << poolType;
            LOG(INFO) << "           POOLDIM : " << size;
            LOG(INFO) << "             PAD_H : " << pad_h;
            LOG(INFO) << "             PAD_W : " << pad_w;
            LOG(INFO) << "          STRIDE_H : " << stride_h;
            LOG(INFO) << "          STRIDE_W : " << stride_w;

        }else if(string("HIDDEN") == type)
        {
            int NumHidden = getIntVariable(vStrLayers[i], "NUM_NEURONS");
            float init_w = getFloatVariable(vStrLayers[i], "INIT_W");
            float lrate = getFloatVariable(vStrLayers[i], "LEARN_RATE");
            float weight_decay = getFloatVariable(vStrLayers[i], "WEIGHT_DECAY");

            layer = new HiddenLayerConfig(type, name, input, sub_input, NumHidden, init_w, lrate, weight_decay );

            cout << endl ;
            LOG(INFO) <<"***********************Hidden layer********************";
            cout << endl;
            LOG(INFO) <<"              NAME : " << name;
            LOG(INFO) <<"             INPUT : " << input;
            LOG(INFO) <<"         SUB_INPUT : " << sub_input;
            LOG(INFO) <<"       NUM_NEURONS : " << NumHidden;
            LOG(INFO) <<"            INIT_W : " << init_w;
            LOG(INFO) <<"        LEARN_RATE : " << lrate;
            LOG(INFO) <<"      WEIGHT_DECAY : " << weight_decay;

        }else if(string("SOFTMAX") == type)
        {
            int nclasses = getIntVariable(vStrLayers[i], "NUM_CLASSES");
            float weight_decay = getFloatVariable(vStrLayers[i], "WEIGHT_DECAY");
            layer = new SoftMaxLayerConfig(type, name , input, sub_input, nclasses, weight_decay);

            cout << endl ;
            LOG(INFO)<<"***********************SoftMax layer*******************";
            cout << endl;
            LOG(INFO) <<"              NAME : " << name;
            LOG(INFO) <<"             INPUT : " << input;
            LOG(INFO) <<"         SUB_INPUT : " << sub_input;
            LOG(INFO) <<"       NUM_CLASSES : " << nclasses;
            LOG(INFO) <<"      WEIGHT_DECAY : " << weight_decay;
            cout << endl<<endl;

        }else if(string("DATA") == type)
        {
            string isTransformer = getStringVariable(vStrLayers[i], "DATA_TRANSFORMER");
            int cropSize = getIntVariable(vStrLayers[i], "CROP_SIZE");
            string mirror = getStringVariable(vStrLayers[i], "DO_MIRROR");
            float scale = getFloatVariable(vStrLayers[i], "SCALE");
            layer = new DataLayerConfig(type ,name, input, sub_input, isTransformer, cropSize, mirror, scale);
            LOG(INFO) << endl ;
            LOG(INFO) <<"***********************Data layer**********************";
            cout<< endl;
            LOG(INFO) <<"              NAME : " << name;
            LOG(INFO) <<"  DATA_TRANSFORMER : " << isTransformer;
            if(string("TRUE") ==isTransformer)
            {
            LOG(INFO) <<"         CROP_SIZE : "<< cropSize;    
            LOG(INFO) <<"         DO_MIRROR : "<< mirror;    
            LOG(INFO) <<"             SCALE : "<< scale;    
            }

        }else if(type == string("ACTIVATION"))
        {
            string non_linearity = getStringVariable(vStrLayers[i], "NON_LINEARITY");
            m_nonLinearConfig = new ConfigNonLinearity(non_linearity);
            layer = new ActivationLayerConfig(type, name, input, sub_input, m_nonLinearConfig->getNonLinearType());

            cout << endl;
            LOG(INFO) <<"********************Activation layer*******************";
            cout << endl;
            LOG(INFO) <<"              NAME : " << name;
            LOG(INFO) <<"             INPUT : " << input;
            LOG(INFO) <<"        SUB_INPUT  : " << sub_input;
            LOG(INFO) <<"     NON_LINEARITY : " << non_linearity;

        }else if(string("LRN") == type)
        {
            unsigned lrnN = getIntVariable(vStrLayers[i],"LRNN");
            float lrnAlpha = getFloatVariable(vStrLayers[i], "LRNALPHA");
            float lrnBeta = getFloatVariable(vStrLayers[i], "LRNBETA");

            layer = new LRNLayerConfig(type, name, input, sub_input, lrnN, lrnAlpha, lrnBeta);

            cout << endl;
            LOG(INFO) << "***********************LRN layer**********************";
            cout << endl;
            LOG(INFO) <<"               NAME : " << name;
            LOG(INFO) <<"              INPUT : " << input;
            LOG(INFO) <<"          SUB_INPUT : " << sub_input;
            LOG(INFO) <<"               LRNN : " << lrnN;
            LOG(INFO) <<"           LRNALPHA : " << lrnAlpha;
            LOG(INFO) <<"            LRNBETA : " << lrnBeta;

        }else if(string("INCEPTION") == type)
        {
            int one = getIntVariable(vStrLayers[i], "ONE");
            int three = getIntVariable(vStrLayers[i], "THREE");
            int five = getIntVariable(vStrLayers[i], "FIVE");
            int three_reduce = getIntVariable(vStrLayers[i], "THREE_REDUCE");
            int five_reduce = getIntVariable(vStrLayers[i], "FIVE_REDUCE");
            int pool_proj = getIntVariable(vStrLayers[i], "POOL_PROJ");
            float init_w = getFloatVariable(vStrLayers[i], "INIT_W");
            float lrate = getFloatVariable(vStrLayers[i], "LEARN_RATE");
            float weight_decay = getFloatVariable(vStrLayers[i], "WEIGHT_DECAY");

            layer = new InceptionLayerConfig(type, name, input, sub_input, one, three, five, three_reduce, five_reduce,
                                        pool_proj, init_w, lrate, weight_decay);
            cout << endl;
            LOG(INFO) <<"********************Inception layer*******************";
            cout << endl;
            LOG(INFO) <<"              NAME : " << name;
            LOG(INFO) <<"             INPUT : " << input;
            LOG(INFO) <<"         SUB_INPUT : " << sub_input;
            LOG(INFO) <<"              ONE  : " << one;
            LOG(INFO) <<"             THREE : " << three;
            LOG(INFO) <<"              FIVE : " << five;
            LOG(INFO) <<"      THREE_REDUCE : " << three_reduce;
            LOG(INFO) <<"       FIVE_REDUCE : " << five_reduce;
            LOG(INFO) <<"         POOL_PROJ : " << pool_proj;
            LOG(INFO) <<"            INIT_W : " << init_w;
            LOG(INFO) <<"        LEARN_RATE : " << lrate;
            LOG(INFO) <<"      WEIGHT_DECAY : " << weight_decay;

        }else if(string("DROPOUT") == type)
        {
            float rate = getFloatVariable(vStrLayers[i], "DROP_RATE");
            layer = new DropOutLayerConfig(type, name, input, sub_input, rate);
            cout << endl;
            LOG(INFO) <<"*********************DropOut layer********************";
            LOG(INFO) <<"              NAME : " << name;
            LOG(INFO) <<"             INPUT : " << input;
            LOG(INFO) <<"         SUB_INPUT : " << sub_input;
            LOG(INFO) <<"         DROP_RATE : " << rate;

        }else if(string("BRANCH") == type)
        {
            vector<string> outputs = getVectorVariable(vStrLayers[i], "OUTPUTS");
            layer = new BranchLayerConfig(type, name, input, sub_input, outputs);
            cout << endl;
            LOG(INFO) <<"***********************Branch layer********************";
            cout << endl;
            LOG(INFO) <<"              NAME : " << name;
            LOG(INFO) <<"             INPUT : " << input;
            LOG(INFO) <<"         SUB_INPUT : " << sub_input;
            LOG(INFO) <<"            OUTPUT : ";
            for(int n = 0; n < outputs.size(); n++)
            {
                LOG(INFO)<< outputs[n]<<" ";
            }
        cout<< endl;
        }
        
        layersInsertByName(name, layer);

        if(std::string("DATA") == type){
            m_firstLayer = layer;
        }else
        {
            /*link the point*/
            m_mapLayers[layer->getInput()]->mutable_vecNext().push_back(layer);
            m_mapLayers[name]->mutable_vecPrev().push_back(m_mapLayers[layer->getInput()]);
        }

        if(std::string("SOFTMAX") == type)
        {
            m_lastLayer = layer;
        }
    }
}
