#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include "./eigen-3.4.0/Eigen/Dense"


#ifndef _UT_
#define _UT_

using namespace std;
using namespace Eigen;


float UT_lambda = 3;        // 调节个样本点权重


// sigma点
template<
typename State_type,
typename StateMat_type,
typename CovMat_type
>
void ut_sigma_sample(
    const State_type& state,
    const CovMat_type& P,
    StateMat_type& state_mat
){
    const int dims = state.rows();
    const int samples_nums = dims*2 + 1;
    // 矩阵分解
    LLT<CovMat_type> llt((dims*1.0 + UT_lambda) * P);
    CovMat_type L = llt.matrixL();
    State_type old_state_s = state;
    state_mat.col(0) = state;
    for(int i=0;i<dims;++i){
        old_state_s = state + L.col(i);
        state_mat.col(i+1) = old_state_s;
    }
    for(int i=0;i<dims;++i){
        old_state_s = state - L.col(i);
        state_mat.col(dims+i+1) = old_state_s;
    }
}


// 样本权重
template<
typename Weight_type
>
void ut_w(const int& dims, Weight_type& w){
    const int samples_nums = dims*2 + 1;
    // 权重
    w(0) = UT_lambda / (dims + UT_lambda);
    for(int i=0;i<samples_nums-1;i++){
        w(i+1) = 1 / (2*(dims + UT_lambda));
    }
}


template<
typename StateMat_type,
typename State_type,
typename CovMat_type,

typename NewStateMat_type,
typename NewState_type,
typename NewCovMat_type
>
void ut(
    const State_type& old_state,     // 原状态
    const CovMat_type& old_P,        // 原状态协方差矩阵
    NewState_type (*forward_function)(const State_type&),    // 转移函数
    StateMat_type& old_state_mat,       // 原状态样本矩阵
    NewStateMat_type& new_state_mat,    // 新状态样本矩阵
    NewState_type& new_state_mean,      // 新状态平均值
    NewCovMat_type& new_P               // 新状态协方差矩阵
    ){
    
    const int dims = old_state.rows();
    const int samples_nums = dims*2 + 1;
    if(
        samples_nums != new_state_mat.cols() || 
        new_state_mean.rows() != new_state_mat.rows() || 
        new_P.rows() != new_state_mat.rows()
    ){
        throw std::invalid_argument("new_state_mat shape error");
    }

    // 样本权重
    Vector<float, samples_nums> w;
    ut_w(dims, w);

    // sigma点
    ut_sigma_sample(old_state, old_P, old_state_mat);
    
    // 状态转移
    NewState_type new_state = forward_function(old_state_mat.col(0));
    new_state_mat.col(0) = new_state;
    for(int i=0;i<dims;++i){
        new_state = forward_function(old_state_mat.col(i+1));
        new_state_mat.col(i+1) = new_state;
    }
    for(int i=0;i<dims;++i){
        new_state = forward_function(old_state_mat.col(dims+i+1));
        new_state_mat.col(dims+i+1) = new_state;
    }

    // 转移后的均值与协方差矩阵
    new_state_mean = new_state_mat * w;
    new_P.setZero();
    for(int i=0;i<samples_nums;++i){
        NewState_type ti = new_state_mat.col(i) - new_state_mean;
        float wi = w(i);
        new_P = new_P + wi * ti * ti.transpose();
    }
}


#endif








