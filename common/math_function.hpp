/*************************************************************************
	> File Name: math_function.hpp
	> Author: TDX 
	> Mail: SA614149@mail.ustc.edu.cn
	> Created Time: 2016年12月12日 星期一 14时12分20秒
 ************************************************************************/

#ifndef _MATH_FUNCTION_HPP_
#define _MATH_FUNCTION_HPP_

#include<cblas.h>
#include<cublas_v2.h>

inline void cblas_saxpby(const int N, const float alpha, const float* X,
                         const int incX, const float beta, float* Y, 
                         const int incY)
{
    cblas_sscal(N, beta, Y, incY);
    cblas_saxpy(N, alpha, X, incX, Y, incY);
}

inline void cblas_daxpby(const int N, const double alpha, const double* X,
                         const int incX, const double beta, double* Y,
                         const int incY)
{
    cblas_dscal(N, beta, Y, incY);
    cblas_daxpy(N, alpha, X, incX, Y, incY);
}

// cpu: y = x * alpha + y
template<typename Ntype>
void cpu_axpy(const int N, const Ntype alpha, const Ntype* X, Ntype* Y);

// cpu: Y = alpha * X + beta * Y
template<typename Ntype>
void cpu_axpby(const int N, const Ntype alpha, const Ntype* X, const Ntype beta, Ntype* Y);

// cpu: x = x * alpha
template<typename Ntype>
void cpu_scal(const int N, const Ntype alpha, Ntype* X);

#endif
