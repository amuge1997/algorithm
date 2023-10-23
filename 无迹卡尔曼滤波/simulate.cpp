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
typename State,
typename TransNoise,

typename Meansure, 
typename MeansureTrain,
typename MeansureNoise
>
class Simulate{
    public:
    Simulate(
        const State& init_x,
        const TransNoise& Q,
        const MeansureTrain& H,
        const MeansureNoise& R,
        State (*forward_function)(const State&)
    ){
        this->x_real.push_back(init_x);
        this->Q = Q;
        this->H = H;
        this->R = R;
        this->forward_function = forward_function;
    }

    vector<State> get_x_real(){
        return this->x_real;
    }

    State x_start(){
        State now_x = this->x_real[this->x_real.size() - 1];
        return now_x;
    }

    // 状态步进
    State x_step(){
        State last_x = this->x_real[this->x_real.size() - 1];
        State now_x_temp = this->forward_function(last_x);
        // 生成符合过程噪声协方差矩阵的噪声向量
        normal_distribution<float> distribution(0.0, 1.0);
        State noise;
        for (int i = 0; i < noise.size(); ++i) {
            noise(i) = distribution(this->generator);
        }
        LLT<TransNoise> llt(this->Q);
        TransNoise M = llt.matrixL();
        State now_x =  M * noise + now_x_temp;
        this->x_real.push_back(now_x);
        return now_x;
    }

    // 观测步进
    Meansure z_step(){
        State now_x_temp = this->x_real[this->x_real.size() - 1];
        Meansure now_z_temp = this->H * now_x_temp;

        normal_distribution<float> distribution(0.0, 1.0);
        Meansure noise;
        for (int i = 0; i < noise.size(); ++i) {
            noise(i) = distribution(this->generator);
        }
        LLT<MeansureNoise> llt(this->R);
        MeansureNoise M = llt.matrixL();
        Meansure now_z_noise = M * noise + now_z_temp;

        this->z_real.push_back(now_z_temp);
        this->z_noise.push_back(now_z_noise);

        return now_z_noise;
    }

    private:
    vector<State> x_real;
    vector<Meansure> z_real;
    vector<Meansure> z_noise;

    TransNoise Q;

    MeansureTrain H;
    MeansureNoise R;

    State (*forward_function)(const State&);

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






