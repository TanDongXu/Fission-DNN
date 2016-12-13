/*************************************************************************
	> File Name: math_function.hpp
	> Author: TDX 
	> Mail: SA614149@mail.ustc.edu.cn
	> Created Time: 2016年12月12日 星期一 14时12分20秒
 ************************************************************************/

#include"math_function.hpp"

// cpu: y = x * alpha + y
template<>
void cpu_axpy<float>(const int N, const float alpha, const float* X, float* Y)
{
    cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template<>
void cpu_axpy<double>(const int N, const double alpha, const double* X, double* Y)
{
    cblas_daxpy(N, alpha, X, 1, Y, 1);
}

// cpu: Y = alpha * X + beta * Y
template<>
void cpu_axpby<float>(const int N, const float alpha, const float* X, const float beta, float* Y)
{
    cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template<>
void cpu_axpby<double>(const int N, const double alpha, const double* X, const double beta, double* Y)
{
   cblas_daxpby(N, alpha, X, 1, beta, Y, 1); 
}

// cpu: x = x * alpha
template<>
void cpu_scal<float>(const int N, const float alpha, float* X)
{
    cblas_sscal(N, alpha, X, 1);
}

template<>
void cpu_scal<double>(const int N, const double alpha, double* X)
{
    cblas_dscal(N, alpha, X, 1);
}

