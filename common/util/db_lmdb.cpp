/*************************************************************************
	> File Name: db_lmdb.cpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月15日 星期四 15时13分55秒
 ************************************************************************/

#include<iostream>
#include<string>
#include<sys/stat.h>
#include"db_lmdb.hpp"

using namespace std;

void LMDB::Open(const string& source, Mode mode) 
{
    MDB_CHECK(mdb_env_create(&m_mdb_env));//创建mdb操作环境
    if (mode == NEW) 
    {
        CHECK_EQ(mkdir(source.c_str(), 0744), 0) << "mkdir " << source << " failed";
    }
    int flags = 0;
    if (mode == READ) {
        flags = MDB_RDONLY | MDB_NOTLS;
    }
    int rc = mdb_env_open(m_mdb_env, source.c_str(), flags, 0664);//打开创建的环境
#ifndef ALLOW_LMDB_NOLOCK
  MDB_CHECK(rc);
#else
    if (rc == EACCES) 
    {
        LOG(WARNING) << "Permission denied. Trying with MDB_NOLOCK ...";
        // Close and re-open environment handle
        mdb_env_close(m_mdb_env);
        MDB_CHECK(mdb_env_create(&m_mdb_env));
        // Try again with MDB_NOLOCK
        flags |= MDB_NOLOCK;
        MDB_CHECK(mdb_env_open(m_mdb_env, source.c_str(), flags, 0664));
    } else 
    {
        MDB_CHECK(rc);
    }
#endif
    LOG(INFO) << "Opened lmdb " << source;
}

LMDBCursor* LMDB::NewCursor() 
{
    MDB_txn* mdb_txn;
    MDB_cursor* mdb_cursor;
    MDB_CHECK(mdb_txn_begin(m_mdb_env, NULL, MDB_RDONLY, &mdb_txn));
    MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &m_mdb_dbi));
    MDB_CHECK(mdb_cursor_open(mdb_txn, m_mdb_dbi, &mdb_cursor));
    return new LMDBCursor(mdb_txn, mdb_cursor);
}

LMDBTransaction* LMDB::NewTransaction() 
{
    return new LMDBTransaction(m_mdb_env);
}

void LMDBTransaction::Put(const string& key, const string& value) 
{
    m_keys.push_back(key);
    m_values.push_back(value);
}

void LMDBTransaction::Commit() 
{
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_data;
    MDB_txn* mdb_txn;

    // Initialize MDB variables
    MDB_CHECK(mdb_txn_begin(m_mdb_env, NULL, 0, &mdb_txn));
    MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi));

    for (int i = 0; i < m_keys.size(); i++) 
    {
        mdb_key.mv_size =m_keys[i].size();
        mdb_key.mv_data = const_cast<char*>(m_keys[i].data());
        mdb_data.mv_size = m_values[i].size();
        mdb_data.mv_data = const_cast<char*>(m_values[i].data());

        // Add data to the transaction
        int put_rc = mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);
        if (put_rc == MDB_MAP_FULL) 
        {
            // Out of memory - double the map size and retry
            mdb_txn_abort(mdb_txn);
            mdb_dbi_close(m_mdb_env, mdb_dbi);
            DoubleMapSize();
            Commit();
            return;
        }
    // May have failed for some other reason
    MDB_CHECK(put_rc);
    }

    // Commit the transaction
    int commit_rc = mdb_txn_commit(mdb_txn);
    if (commit_rc == MDB_MAP_FULL) 
    {
        // Out of memory - double the map size and retry
        mdb_dbi_close(m_mdb_env, mdb_dbi);
        DoubleMapSize();
        Commit();
        return;
    }
    // May have failed for some other reason
    MDB_CHECK(commit_rc);

    // Cleanup after successful commit
    mdb_dbi_close(m_mdb_env, mdb_dbi);
    m_keys.clear();
    m_values.clear();
}

void LMDBTransaction::DoubleMapSize() 
{
    struct MDB_envinfo current_info;
    MDB_CHECK(mdb_env_info(m_mdb_env, &current_info));
    size_t new_size = current_info.me_mapsize * 2;
    DLOG(INFO) << "Doubling LMDB map size to " << (new_size>>20) << "MB ...";
    MDB_CHECK(mdb_env_set_mapsize(m_mdb_env, new_size));
}
