#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include "./eigen-3.4.0/Eigen/Dense"


#ifndef _SIM_
#define _SIM_

using namespace std;
using namespace Eigen;

// 仿真环境
template<
typename X_type,
typename Trans_Noise_type,

typename Z_type, 
typename H_type,
typename Meansure_Noise_type
>
class Simulate{
    public:
    Simulate(
        const X_type& init_x,
        const Trans_Noise_type& Q,
        const H_type& H,
        const Meansure_Noise_type& R,
        X_type (*forward_function)(const X_type&)
    ){
        this->x_real.push_back(init_x);
        this->Q = Q;
        this->H = H;
        this->R = R;
        this->forward_function = forward_function;
    }

    vector<X_type> get_x_real(){
        return this->x_real;
    }

    X_type x_start(){
        X_type now_x = this->x_real[this->x_real.size() - 1];
        return now_x;
    }

    // 状态步进
    X_type x_step(){
        X_type last_x = this->x_real[this->x_real.size() - 1];
        X_type now_x_temp = this->forward_function(last_x);
        // 生成符合过程噪声协方差矩阵的噪声向量
        normal_distribution<float> distribution(0.0, 1.0);
        X_type noise;
        for (int i = 0; i < noise.size(); ++i) {
            noise(i) = distribution(this->generator);
        }
        LLT<Trans_Noise_type> llt(this->Q);
        Trans_Noise_type M = llt.matrixL();
        X_type now_x =  M * noise + now_x_temp;
        this->x_real.push_back(now_x);
        return now_x;
    }

    // 观测步进
    Z_type z_step(){
        X_type now_x_temp = this->x_real[this->x_real.size() - 1];
        Z_type now_z_temp = this->H * now_x_temp;

        normal_distribution<float> distribution(0.0, 1.0);
        Z_type noise;
        for (int i = 0; i < noise.size(); ++i) {
            noise(i) = distribution(this->generator);
        }
        LLT<Meansure_Noise_type> llt(this->R);
        Meansure_Noise_type M = llt.matrixL();
        Z_type now_z_noise = M * noise + now_z_temp;

        this->z_real.push_back(now_z_temp);
        this->z_noise.push_back(now_z_noise);

        return now_z_noise;
    }

    private:
    vector<X_type> x_real;
    vector<Z_type> z_real;
    vector<Z_type> z_noise;

    Trans_Noise_type Q;

    H_type H;
    Meansure_Noise_type R;

    X_type (*forward_function)(const X_type&);

    default_random_engine generator;
};

template<
typename T
>
void save(const string& file_name, const T& data){
    ofstream out(file_name);
    if (out.is_open()) {
        for (const VectorXf& vector : data) {
            for (int i = 0; i < vector.size(); i++) {
                out << vector(i) << "\t";
            }
            out << "\n";
        }
        out.close();
        std::cout << "数据已写入文件 " << file_name << std::endl;
    } else {
        std::cerr << "  无法打开文件 " << file_name << std::endl;
    }
}



#endif






