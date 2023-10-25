#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include "./eigen-3.4.0/Eigen/Dense"
#include "ctrv2.cpp"
#include "simulate.cpp"

using namespace std;
using namespace Eigen;

template<typename T>
void print(const T& t){
    cout<< t << endl;
}


float UT_lambda = 1;        // 调节个样本点权重


// sigma点
template<
typename State,
typename StateMat,
typename CovMat
>
void ut_sigma_sample(
    const State& state,
    const CovMat& P,
    StateMat& state_mat
){
    const int dims = state.rows();
    const int samples_nums = dims*2 + 1;
    // 矩阵分解
    LLT<CovMat> llt((dims*1.0 + UT_lambda) * P);
    CovMat L = llt.matrixL();
    State old_state_s = state;
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

// UT变换
template<
typename StateMat,
typename State,
typename CovMat,

typename NewStateMat,
typename NewState,
typename NewCovMat
>
void ut(
    const State& old_state,     // 原状态
    const CovMat& old_P,        // 原状态协方差矩阵
    StateMat& old_state_mat,       // 原状态样本矩阵
    NewState (*forward_function)(const State&),    // 转移函数
    NewStateMat& new_state_mat,    // 新状态样本矩阵
    NewState& new_state_mean,      // 新状态平均值
    NewCovMat& new_P               // 新状态协方差矩阵
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
    NewState new_state = forward_function(old_state_mat.col(0));
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
        NewState ti = new_state_mat.col(i) - new_state_mean;
        float wi = w(i);
        new_P = new_P + wi * ti * ti.transpose();
    }
}


// 无迹卡尔曼
template<
typename State, 
typename TransNoise,

typename Meansure,
typename MeansureNoise
>
tuple<State, TransNoise, State> UnscentedKalmanFilter(
    const State& old_x,             // 当前状态估计
    const TransNoise& old_P,        // 状态协方差矩阵
    const TransNoise& Q,            // 过程噪声协方差矩阵

    const Meansure& now_z,          // 当前测量值
    const MeansureNoise& R,         // 测量噪声协方差矩阵

    State (*forward_function)(const State&),        // 状态转移函数
    Meansure (*meansure_function)(const State&)     // 观测函数
){  
    const int dims = old_x.rows();
    const int samples_nums = 2*dims + 1;

    // 预测步骤
    State predict_x_noise_mean;
    Matrix<float, old_x.rows(), samples_nums> old_x_mat;
    Matrix<float, predict_x_noise_mean.rows(), samples_nums> predict_x_mat;
    TransNoise predict_x_noise_P;
    ut( 
        old_x, 
        old_P, 
        old_x_mat, 
        forward_function, 
        predict_x_mat, 
        predict_x_noise_mean, 
        predict_x_noise_P
        );
    predict_x_noise_P = predict_x_noise_P + Q;

    // 观测步骤
    Meansure x2z_mean;
    Matrix<float, predict_x_noise_mean.rows(), samples_nums> predict_x_noise_ut_sample_mat;
    Matrix<float, x2z_mean.rows(), samples_nums> x2z_mat;
    MeansureNoise x2z_P;
    ut(
        predict_x_noise_mean, 
        predict_x_noise_P, 
        predict_x_noise_ut_sample_mat, 
        meansure_function,
        x2z_mat,
        x2z_mean,
        x2z_P
        );
    MeansureNoise S = x2z_P + R;

    Vector<float, samples_nums> w;
    ut_w(dims, w);
    Matrix<float, predict_x_noise_mean.rows(), samples_nums> sigma_x;
    Matrix<float, x2z_mean.rows(), samples_nums> sigma_z;
    sigma_x = predict_x_noise_ut_sample_mat.colwise() - predict_x_noise_mean;
    // Sigma_x = predict_x_mat.colwise() - predict_x_noise_mean;
    sigma_z = x2z_mat.colwise() - x2z_mean;
    Matrix<float, predict_x_noise_mean.rows(), x2z_mean.rows()> sigma;
    sigma.setZero();
    for(int i=0;i<samples_nums;++i){
        sigma = sigma + w(i) * sigma_x.col(i) * sigma_z.col(i).transpose();
    }
    
    // 卡尔曼增益
    Matrix<float, predict_x_noise_mean.rows(), x2z_mean.rows()> K;
    K = sigma * S.inverse();

    // 融合
    State new_x = predict_x_noise_mean + K * (now_z - x2z_mean);
    TransNoise new_P = predict_x_noise_P - K * S * K.transpose();

    return make_tuple(new_x, new_P, predict_x_noise_mean);
}


// 观测函数
Vector<float, 2> meansure_fuc(const Vector<float, 5>& x){
    Matrix<float, 2, 5> H;
    H <<
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0
    ;
    Vector<float, 2> z = H * x;
    return z;
}

void run(){
    typedef Matrix<float, 5, 5> mat5x5f;
    typedef Matrix<float, 2, 2> mat2x2f;
    typedef Matrix<float, 2, 5> mat2x5f;
    typedef Vector<float, 5> vec5f;
    typedef Vector<float, 2> vec2f;

    // 初始状态
    vec5f init_x;
    init_x <<0, 0, 0.5, 0, 10 * M_PI/180;

    // 状态误差
    mat5x5f P = mat5x5f::Zero();

    // 状态转移噪声
    mat5x5f Q;
    Q <<
        0.00001, 0, 0, 0, 0,
        0, 0.00001, 0, 0, 0,
        0, 0, 0.001, 0, 0,
        0, 0, 0, 0.00001, 0,
        0, 0, 0, 0, 0.0005
    ;
    
    mat2x5f H;
    H <<
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0
    ;
    // 观测误差
    mat2x2f R;
    R <<
        0.01, 0,
        0, 0.01
    ;

    // ctrv仿真环境
    Simulate<
    vec5f,
    mat5x5f,
    vec2f,
    mat2x5f,
    mat2x2f
    > sim(init_x, Q, H, R, ctrv_forward);

    // UKF
    vec5f old_x = sim.x_start();
    mat5x5f old_P = P;
    vector<vec5f> predict_x_record;
    vector<vec5f> now_x_record;
    vector<vec2f> now_z_record;
    for(int i=0;i<60;++i){
        sim.x_step();
        vec2f now_z_noise = sim.z_step();

        auto result = UnscentedKalmanFilter(old_x, old_P, Q, now_z_noise, R, ctrv_forward, meansure_fuc);

        vec5f now_x = get<0>(result);
        mat5x5f now_P = get<1>(result);
        vec5f now_predict_x = get<2>(result);

        now_x_record.push_back(now_x);
        now_z_record.push_back(now_z_noise);
        predict_x_record.push_back(now_predict_x);
        
        old_x = now_x;
        old_P = now_P;
    }

    // 保存结果
    vector<vec5f> x_real_ = sim.get_x_real();
    vector<vec5f> x_real(x_real_.begin()+1, x_real_.end());
    save("now_real_x.txt", x_real);
    save("predict_x.txt", predict_x_record);
    save("now_x.txt", now_x_record);
    save("now_z.txt", now_z_record);
    
}


int main(){
    run();
    return 0;
}











