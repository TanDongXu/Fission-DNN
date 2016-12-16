/*************************************************************************
	> File Name: db.hpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月15日 星期四 12时18分39秒
 ************************************************************************/

#ifndef _DB_HPP_
#define _DB_HPP_

#include<string>
#include<iostream>
#include<glog/logging.h>

using namespace std;

enum Mode { READ, WRITE, NEW };

class Cursor
{
    public:
    Cursor(){  }
    virtual ~Cursor(){  };
    virtual void SeekToFirst() = 0;
    virtual void Next() = 0;
    virtual string key() = 0;
    virtual string value() = 0;
    virtual bool valid() = 0;
};

class Transaction
{
    public:
    Transaction(){  }
    virtual ~Transaction(){  } 
    virtual void Put(const string& key, const string& value) = 0;
    virtual void Commit() = 0;
};

class DB
{
    public:
    DB(){  };
    virtual ~DB(){  }
    virtual void Open(const string& source, Mode mode) = 0;
    virtual void Close() = 0;
    virtual Cursor* NewCursor() = 0;
    virtual Transaction* NewTransaction() = 0;
};


DB* GetDB(const string& backend);
//{
//    if("lmdb" == backend)
//    {
//        return new LMDB();
//    }
//
//    LOG(FATAL) << "Unknown dataBase backend" << backend;
//    return NULL;
//}
#endif
