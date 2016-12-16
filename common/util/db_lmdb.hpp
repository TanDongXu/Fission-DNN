/*************************************************************************
	> File Name: db_lmdb.hpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月15日 星期四 14时38分34秒
 ************************************************************************/

#ifndef _DB_LMDB_HPP_
#define _DB_LMDB_HPP_

#include"lmdb.h"
#include"db.hpp"
#include<string>
#include<glog/logging.h>

using namespace std;

inline void MDB_CHECK(int mdb_status)
{
    CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}

class LMDBCursor : public Cursor
{
    public:
    explicit LMDBCursor(MDB_txn* mdb_txn, MDB_cursor* mdb_cursor):
    m_mdb_txn(mdb_txn), m_mdb_cursor(mdb_cursor),m_valid(false)
    {
        SeekToFirst();
    }
    virtual ~LMDBCursor()
    {
        mdb_cursor_close(m_mdb_cursor);
        mdb_txn_abort(m_mdb_txn);
    }
    virtual void SeekToFirst(){ Seek(MDB_FIRST); }
    virtual void Next(){ Seek(MDB_NEXT); }
    virtual string key()
    {
        return string(static_cast<const char*>(m_mdb_key.mv_data), m_mdb_key.mv_size);
    }
    virtual string value()
    {
        return string(static_cast<const char*>(m_mdb_value.mv_data), m_mdb_value.mv_size);
    }
    virtual bool valid(){ return m_valid; }

    private:
    void Seek(MDB_cursor_op op)
    {
        int mdb_status = mdb_cursor_get(m_mdb_cursor, &m_mdb_key, &m_mdb_value, op);
        if(MDB_NOTFOUND == mdb_status)
        {
            m_valid = false;
        }else
        {
            MDB_CHECK(mdb_status);
            m_valid = true;
        }
    }

    MDB_txn* m_mdb_txn;
    MDB_cursor* m_mdb_cursor;
    MDB_val m_mdb_key, m_mdb_value;
    bool m_valid;
};

class LMDBTransaction : public Transaction
{
    public:
    explicit LMDBTransaction(MDB_env* mdb_env):m_mdb_env(mdb_env) {}
    virtual void Put(const string& key, const string& value);
    virtual void Commit();

    private:
    MDB_env* m_mdb_env;
    vector<string> m_keys, m_values;
    void DoubleMapSize();
};

class LMDB : public DB
{
    public:
    LMDB() : m_mdb_env(NULL) {  }
    virtual ~LMDB() { Close();}
    virtual void Open(const string& source, Mode mode);
    virtual void Close()
    {
        if(NULL != m_mdb_env)
        {
            mdb_dbi_close(m_mdb_env, m_mdb_dbi);
            mdb_env_close(m_mdb_env);
            m_mdb_env = NULL;
        }
    }

    virtual LMDBCursor* NewCursor();
    virtual LMDBTransaction* NewTransaction();

    private:
    MDB_env* m_mdb_env;
    MDB_dbi m_mdb_dbi;
};

#endif
