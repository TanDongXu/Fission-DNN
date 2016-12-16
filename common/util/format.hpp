/*************************************************************************
	> File Name: format.hpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2016年12月15日 星期四 21时29分09秒
 ************************************************************************/

#ifndef _FORMAT_HPP_
#define _FORMAT_HPP_

#include<iostream>
#include<sstream>
#include<string>
#include<iomanip>
using namespace std;

inline string format_int(int n, int numberOfLeadingZeros = 0)
{
    ostringstream s;
    s << setw(numberOfLeadingZeros) << setfill('0') << n;
    return s.str();
}
#endif
