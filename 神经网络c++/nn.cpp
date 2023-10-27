#include <iostream>
#include <vector>
#include <random>
#include "./eigen-3.4.0/Eigen/Dense"
#include "nn_utils.cpp"


using namespace std;
using namespace Eigen;


// 网络
class NN{
    public:
    
    NN(int x_dims, int y_dims, int hidden_dims, float lr, int epochs, int batch){
        this->x_dims = x_dims;              // 输入维度
        this->y_dims = y_dims;              // 输出维度
        this->hidden_dims = hidden_dims;    // 中间维度
        this->lr = lr;                      // 学习率
        this->epochs = epochs;              // 轮次
        this->batch = batch;                // 批数量
        this->weights1.resize(hidden_dims, x_dims+1);   // 参数矩阵1
        this->weights2.resize(y_dims, hidden_dims+1);   // 参数矩阵2
    }

    // 预测
    MatrixXf predict(const MatrixXf& x){
        int nums = x.cols();
        
        MatrixXf bias(1, nums);                         // 偏置向量
        MatrixXf x_bias(x_dims + 1, nums);               // 添加偏置后的输入矩阵
        MatrixXf hidden_sig_bias(hidden_dims+1, nums);   // 添加偏置后的中间矩阵

        MatrixXf hidden;
        MatrixXf hidden_sig;
        MatrixXf out;
        
        bias.setOnes();
        x_bias << x, bias;
        hidden = weights1 * x_bias;
        hidden_sig = 1.0 / (1.0 + (-hidden.array()).exp());
        hidden_sig_bias << hidden_sig, bias;
        
        out = weights2 * hidden_sig_bias;
        return out;
    }

    // 训练
    void train(const MatrixXf& x_ori, const MatrixXf& y_ori){
        int nums = x_ori.cols();            // 样本数量
        
        MatrixXf bias(1, batch);            // 偏置向量
        MatrixXf x(x_dims, batch);          // 输入层
        MatrixXf y(y_dims, batch);          // 标签
        MatrixXf x_bias(x_dims + 1, batch);  // 输入偏置层
        MatrixXf hidden_sig_bias(hidden_dims+1, batch);  // 中间偏置层

        MatrixXf hidden;                // 中间层
        MatrixXf hidden_sig;            // sigmoid激活层
        MatrixXf out;                   // 输出层
        MatrixXf error;                 // 误差
        float error2;                   // 总误差
        MatrixXf dEdW2;                 // 误差对W2梯度矩阵
        MatrixXf dEdHsig;
        MatrixXf dEdHone;
        MatrixXf dEdH;                  // 误差对中间层梯度矩阵
        MatrixXf dEdW1;                 // 误差对W1梯度矩阵
        
        // 初始化参数矩阵
        weights1 = MatrixXf::Random(hidden_dims, x_dims+1).array() / sqrt(hidden_dims + x_dims+1);
        weights2 = MatrixXf::Random(y_dims, hidden_dims+1).array() / sqrt(y_dims + hidden_dims+1);
        // 偏置向量
        bias.setOnes();
        
        // 序号列表 用于样本批次抽样
        std::srand(unsigned(std::time(nullptr)));
        vector<int> numbers;
        for (int i = 0; i < nums; i++) {
            numbers.push_back(i);
        }
        for(int i=0;i<epochs;i++){
            // 随机抽样
            random_shuffle(numbers.begin(), numbers.end());
            for (int i = 0; i < batch; i++) {
                x.col(i) = x_ori.col(numbers[i]);
                y.col(i) = y_ori.col(numbers[i]);
            }
            // 添加偏置
            x_bias << x, bias;
            
            // h = sigmoid( w1 * x )    使用sigmoid激活
            hidden = weights1 * x_bias;
            hidden_sig = 1.0 / (1.0 + (-hidden.array()).exp());
            hidden_sig_bias << hidden_sig, bias;

            // 为了能全域映射,不进行激活直接线性映射得到输出. 由于输出层不进行激活,因此该网络的非线性能力不强
            // o = w2 * h               
            out = weights2 * hidden_sig_bias;

            // 误差
            // e = (Y - O)^2
            error = y - out;
            error2 = error.array().square().sum();
            
            // 链式法则求各矩阵梯度
            dEdW2 = -2. * (error * hidden_sig_bias.transpose()).array();
            dEdHsig = -2. * (weights2.transpose() * error).array();
            dEdHone = hidden_sig_bias.array() * (1.0 - hidden_sig_bias.array());
            dEdH = dEdHone.topRows(dEdHone.rows()-1);
            dEdW1 = dEdH * x_bias.transpose();
            
            // 更新参数矩阵
            weights1 = weights1 - lr * dEdW1;
            weights2 = weights2 - lr * dEdW2;
        }
    }

    private:
    int x_dims;
    int y_dims;
    int hidden_dims;
    float lr;
    int epochs;
    int batch;
    MatrixXf weights1;
    MatrixXf weights2;
};


int main(){

    MatrixXf xt = readx("x.txt").transpose();
    MatrixXf yt = ready("y.txt").transpose();
    
    MatrixXf x_mesh = readx("x_mesh.txt").transpose();

    int x_dims = xt.rows();
    int y_dims = yt.rows();
    int hidden_nums = 8;
    float lr = 1e-3;
    int epochs = 1000;
    int batch = 8;

    NN nn(x_dims, y_dims, hidden_nums, lr, epochs, batch);
    nn.train(xt, yt);
    
    MatrixXf p1 = nn.predict(xt);
    MatrixXf predict = nn.predict(x_mesh);

    savey(predict.transpose(), "y_mesh.txt");

    return 0;
}











