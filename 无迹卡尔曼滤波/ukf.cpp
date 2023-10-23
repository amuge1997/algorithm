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











