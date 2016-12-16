/*************************************************************************
	> File Name: db.cpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月16日 星期五 08时38分51秒
 ************************************************************************/

#include<iostream>
#include"db_lmdb.hpp"
#include"db.hpp"

using namespace std;

DB* GetDB(const string& backend)
{
    if("lmdb" == backend)
    {
        return new LMDB();
    }

    LOG(FATAL) << "Unknown dataBase backend "  << backend;
    return NULL;
}

