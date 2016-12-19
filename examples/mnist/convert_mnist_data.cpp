/*************************************************************************
	> File Name: convert_mnist_data.cpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月15日 星期四 10时06分55秒
 ************************************************************************/

#include<iostream>
#include<glog/logging.h>
#include<lmdb.h>
#include<unistd.h>
#include<fstream>
#include<string>
#include<boost/scoped_ptr.hpp>

#include"common/util/db.hpp"
#include"build/datum.pb.h"
#include"common/util/format.hpp"
#include"readData/mnist/data_reader.hpp"


using namespace std;
using namespace caffe;//protobuf create
using boost::scoped_ptr;

void convert_dataSet(const char* image_filename, const char* label_filename,
                     const char* db_path, const string& db_backend)
{
    ifstream image_file(image_filename, ios::in | ios::binary);
    ifstream label_file(label_filename, ios::in | ios::binary);
    CHECK(image_file) << "Unable to open file " << image_filename;
    CHECK(label_file) << "Unable to open file " << label_filename;

    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    CHECK_EQ(num_items, num_labels);
    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    scoped_ptr<DB> db(GetDB(db_backend));
    db->Open(db_path, Mode::NEW);
    scoped_ptr<Transaction> txn(db->NewTransaction());

    //storing to db
    char label;
    char* pixels = new char[rows * cols];
    int count = 0;
    string value;

    Datum datum;
    datum.set_channels(1);
    datum.set_height(rows);
    datum.set_width(cols);
    LOG(INFO) << "A total of " << num_items << " items.";
    LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
    for(int item_id = 0; item_id < num_items; ++item_id)
    {
        image_file.read(pixels, rows * cols);
        label_file.read(&label, 1);
        datum.set_data(pixels);
        datum.set_label(label);
        string key_str = format_int(item_id, 8);
        datum.SerializeToString(&value);

        txn->Put(key_str, value);
        if(++count % 1000 == 0)
        {
            txn->Commit();
        }
    }
    
    if(count % 1000 != 0)
    {
        txn->Commit();
    }

    LOG(INFO) << "Processed " << count << " files.";
    delete[] pixels;
    db->Close();
}

void createMnistLMDB()
{
    const string backend = "lmdb";
    const string data = "data/mnist/";
    const string train_db_path = data + "mnist_train_lmdb";
    const string train_data = data + "train-images-idx3-ubyte";
    const string train_label = data + "train-labels-idx1-ubyte";
    const string test_db_path = data + "mnist_test_lmdb";
    const string test_data = data + "t10k-images-idx3-ubyte";
    const string test_label = data + "t10k-labels-idx1-ubyte";
    // if the file exist, then return 
    if(!access(train_db_path.c_str(), R_OK) || !access(test_db_path.c_str(), R_OK)) return;

    LOG(INFO) << "Createing mnist lmdb ...";
    //convert train set
    convert_dataSet(train_data.c_str(), train_label.c_str(), train_db_path.c_str(), backend);
    //convert test set
    convert_dataSet(test_data.c_str(), test_label.c_str(), test_db_path.c_str(), backend);
    LOG(INFO) << "Done";
}

