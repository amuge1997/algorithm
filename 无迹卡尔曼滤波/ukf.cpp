#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include "./eigen-3.4.0/Eigen/Dense"
#include "ut.cpp"
#include "ctrv2.cpp"
#include "simulate.cpp"

using namespace std;
using namespace Eigen;


template<typename T>
void print(const T& t){
    cout<< t << endl;
}


// 无迹卡尔曼
template<
typename X_type, 
typename Trans_Noise_type,

typename Z_type,
typename Meansure_Noise_type
>
tuple<X_type, Trans_Noise_type, X_type> UnscentedKalmanFilter(
    const X_type& old_x,            // 当前状态估计
    const Trans_Noise_type& P,      // 状态协方差矩阵
    const Trans_Noise_type& Q,      // 过程噪声协方差矩阵

    const Z_type& now_z,            // 当前测量值
    const Meansure_Noise_type& R,   // 测量噪声协方差矩阵

    X_type (*forward_function)(const X_type&),              // 状态转移函数
    Z_type (*meansure_function)(const X_type&)              // 观测函数
){  
    const int dims = old_x.rows();
    const int samples_nums = 2*dims + 1;

    // 预测步骤
    X_type predict_x_noise_mean;
    Matrix<float, old_x.rows(), samples_nums> old_x_mat;
    Matrix<float, predict_x_noise_mean.rows(), samples_nums> predict_x_mat;
    Trans_Noise_type predict_x_noise_P;
    ut( 
        old_x, 
        P, 
        old_x_mat, 
        forward_function, 
        predict_x_mat, 
        predict_x_noise_mean, 
        predict_x_noise_P
        );
    predict_x_noise_P = predict_x_noise_P + Q;

    // 观测步骤
    Z_type predict_x2z_mean;
    Matrix<float, predict_x_noise_mean.rows(), samples_nums> predict_x_noise_ut_sample_mat;
    Matrix<float, predict_x2z_mean.rows(), samples_nums> predict_x2z_mat;
    Meansure_Noise_type predict_x2z_P;
    ut(
        predict_x_noise_mean, 
        predict_x_noise_P, 
        predict_x_noise_ut_sample_mat, 
        meansure_function,
        predict_x2z_mat,
        predict_x2z_mean,
        predict_x2z_P
        );
    Meansure_Noise_type S = predict_x2z_P + R;

    Vector<float, samples_nums> w;
    ut_w(dims, w);
    Matrix<float, predict_x_noise_mean.rows(), samples_nums> Sigma_x;
    Matrix<float, predict_x2z_mean.rows(), samples_nums> Sigma_z;
    Sigma_x = predict_x_noise_ut_sample_mat.colwise() - predict_x_noise_mean;
    // Sigma_x = predict_x_mat.colwise() - predict_x_noise_mean;
    Sigma_z = predict_x2z_mat.colwise() - predict_x2z_mean;
    Matrix<float, predict_x_noise_mean.rows(), predict_x2z_mean.rows()> Sigma;
    Sigma.setZero();
    for(int i=0;i<samples_nums;++i){
        Sigma = Sigma + w(i) * Sigma_x.col(i) * Sigma_z.col(i).transpose();
    }
    
    // 卡尔曼增益
    Matrix<float, predict_x_noise_mean.rows(), predict_x2z_mean.rows()> K;
    K = Sigma * S.inverse();

    // 融合
    X_type new_x = predict_x_noise_mean + K * (now_z - predict_x2z_mean);
    Trans_Noise_type new_P = predict_x_noise_P - K * S * K.transpose();

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











