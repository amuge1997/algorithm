#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include "./eigen-3.4.0/Eigen/Dense"
#include "ctrv2.cpp"

using namespace std;
using namespace Eigen;


template<typename T>
void print(const T& t){
    cout<< t << endl;
}


// 拓展卡尔曼滤波
template<
typename X_type, 
typename Trans_Noise_type,

typename Z_type,
typename H_type,
typename Meansure_Noise_type
>
tuple<X_type, Trans_Noise_type, X_type> ExtendedKalmanFilter(
    const X_type& old_x,            // 当前状态估计
    const Trans_Noise_type& P,      // 状态协方差矩阵
    const Trans_Noise_type& Q,      // 过程噪声协方差矩阵

    const Z_type& now_z,            // 当前测量值
    const H_type& H,                // 测量矩阵
    const Meansure_Noise_type& R,   // 测量噪声协方差矩阵

    X_type (*forward_function)(const X_type&),              // 状态转移函数
    Trans_Noise_type (*jacobian_function)(const X_type&)    // 雅可比矩阵
    )
{
    // 预测
    X_type now_predict_x = forward_function(old_x);
    Trans_Noise_type JA = jacobian_function(old_x);
    
    Trans_Noise_type now_predict_P = JA * P * JA.transpose() + Q;

    // 更新
    Matrix<float, old_x.rows(), now_z.rows()> K;
    K = now_predict_P * H.transpose() * (H * now_predict_P * H.transpose() + R).inverse();
    Trans_Noise_type im;
    im.setIdentity();
    X_type now_x = now_predict_x + K * (now_z - H * now_predict_x);
    Trans_Noise_type now_P = (im - K * H) * now_predict_P;

    return make_tuple(now_x, now_P, now_predict_x);
}


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
    // Q <<
    //     0.00001, 0, 0, 0, 0,
    //     0, 0.00001, 0, 0, 0,
    //     0, 0, 0.00001, 0, 0,
    //     0, 0, 0, 0.00001, 0,
    //     0, 0, 0, 0, 0.000005
    // ;
    // 观测矩阵
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


    // EKF滤波
    vec5f old_x = sim.x_start();
    mat5x5f old_P = P;
    vector<vec5f> predict_x_record;
    vector<vec5f> now_x_record;
    vector<vec2f> now_z_record;
    for(int i=0;i<60;++i){
        sim.x_step();
        vec2f now_z_noise = sim.z_step();

        auto result = ExtendedKalmanFilter(old_x, old_P, Q, now_z_noise, H, R, ctrv_forward, ctrv_jacobian);

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



















