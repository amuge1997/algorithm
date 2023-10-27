#include <iostream>
#include <vector>
#include <random>
#include "./eigen-3.4.0/Eigen/Dense"
#include "nn_utils.cpp"


using namespace std;
using namespace Eigen;


// nn网络
class NN{
    public:
    NN(int x_dims, int y_dims, int hidden_dims, float lr, int epochs, int batch){
        this->x_dims = x_dims;              // 输入维度
        this->y_dims = y_dims;              // 输出维度
        this->hidden_dims = hidden_dims;    // 中间维度
        this->lr = lr;                      // 学习率
        this->epochs = epochs;              // 轮次
        this->batch = batch;                // 批数量
        this->weights1.resize(hidden_dims, x_dims+1);       // 参数矩阵1
        this->weights2.resize(hidden_dims, hidden_dims+1);  // 参数矩阵2
        this->weights3.resize(y_dims, hidden_dims+1);       // 参数矩阵2
    }

    // 预测
    MatrixXf predict(const MatrixXf& x){
        int nums = x.cols();
        
        MatrixXf bias(1, nums);                         // 偏置向量
        MatrixXf x_bias(x_dims + 1, nums);              // 添加偏置后的输入矩阵
        MatrixXf hidden1_sig_bias(hidden_dims+1, nums); // 添加偏置后的中间矩阵
        MatrixXf hidden2_sig_bias(hidden_dims+1, nums); // 添加偏置后的中间矩阵

        MatrixXf hidden1;
        MatrixXf hidden2;
        MatrixXf hidden1_sig;
        MatrixXf hidden2_sig;
        MatrixXf out;
        
        bias.setOnes();
        x_bias << x, bias;

        hidden1 = weights1 * x_bias;
        hidden1_sig = 1.0 / (1.0 + (-hidden1.array()).exp());
        hidden1_sig_bias << hidden1_sig, bias;

        hidden2 = weights2 * hidden1_sig_bias;
        hidden2_sig = 1.0 / (1.0 + (-hidden2.array()).exp());
        hidden2_sig_bias << hidden2_sig, bias;
        
        out = weights3 * hidden2_sig_bias;
        return out;
    }

    // 训练
    void train(const MatrixXf& x_ori, const MatrixXf& y_ori){
        int nums = x_ori.cols();            // 样本数量
        
        MatrixXf bias(1, batch);            // 偏置向量
        MatrixXf x(x_dims, batch);          // 输入层
        MatrixXf y(y_dims, batch);          // 标签
        MatrixXf x_bias(x_dims + 1, batch); // 输入偏置层
        MatrixXf hidden1_sig_bias(hidden_dims+1, batch);    // 中间偏置层
        MatrixXf hidden2_sig_bias(hidden_dims+1, batch);    // 中间偏置层

        MatrixXf hidden1;                   // 隐藏层1
        MatrixXf hidden2;                   // 隐藏层1
        MatrixXf hidden1_sig;               // sigmoid激活层1
        MatrixXf hidden2_sig;               // sigmoid激活层2
        MatrixXf out;                       // 输出层
        MatrixXf error;                     // 误差
        float error2;                       // 总误差

        MatrixXf dEdO;                      // 误差对输出梯度
        MatrixXf dEdW3;                     // 误差对W3梯度

        MatrixXf dEdH2sig;                  // 误差对sigmoid激活层2梯度
        MatrixXf dEdH2bias;                 // 误差对含偏置隐藏层2梯度
        MatrixXf dEdH2;                     // 误差对隐藏层2梯度
        MatrixXf dEdW2;                     // 误差对W2梯度

        MatrixXf dEdH1sig;                  // 误差对sigmoid激活层1梯度
        MatrixXf dEdH1bias;                 // 误差对含偏置隐藏层1梯度
        MatrixXf dEdH1;                     // 误差对隐藏层1梯度
        MatrixXf dEdW1;                     // 误差对W1梯度
        
        // 初始化参数矩阵
        weights1 = MatrixXf::Random(hidden_dims, x_dims+1).array() / sqrt(hidden_dims + x_dims+1);
        weights2 = MatrixXf::Random(hidden_dims, hidden_dims+1).array() / sqrt(hidden_dims + hidden_dims+1);
        weights3 = MatrixXf::Random(y_dims, hidden_dims+1).array() / sqrt(y_dims + hidden_dims+1);

        // 偏置向量
        bias.setOnes();
        
        // 序号列表 用于样本批次抽样
        std::srand(unsigned(std::time(nullptr)));
        vector<int> numbers;
        for (int i = 0; i < nums; i++) {
            numbers.push_back(i);
        }
        for(int i=0;i<epochs;i++){
            // 每批次随机抽样
            random_shuffle(numbers.begin(), numbers.end());
            int batch_sum = nums / batch;
            for(int j=0;j<batch_sum;j++){
                for (int bi = 0; bi < batch; bi++){
                    x.col(bi) = x_ori.col(numbers[bi + j * batch]);
                    y.col(bi) = y_ori.col(numbers[bi + j * batch]);
                }
                
                // 添加偏置
                x_bias << x, bias;
                
                // h1 = sigmoid( w1 * x )    使用sigmoid激活
                hidden1 = weights1 * x_bias;
                hidden1_sig = 1.0 / (1.0 + (-hidden1.array()).exp());
                hidden1_sig_bias << hidden1_sig, bias;

                // h2 = sigmoid( w2 * h1 )    使用sigmoid激活
                hidden2 = weights2 * hidden1_sig_bias;
                hidden2_sig = 1.0 / (1.0 + (-hidden2.array()).exp());
                hidden2_sig_bias << hidden2_sig, bias;
                
                // 为了能全域映射,不进行激活直接线性映射得到输出
                // o = w3 * h2
                out = weights3 * hidden2_sig_bias;

                // 误差
                // e = (Y - O)^2
                error = y - out;
                error2 = error.array().square().sum() / batch;
                
                // 链式法则求各矩阵梯度
                // 输出层梯度
                dEdO = -2./batch * error.array();
                dEdW3 = dEdO * hidden2_sig_bias.transpose();
                dEdH2sig = weights3.transpose() * dEdO;

                // 第二层梯度
                dEdH2bias = dEdH2sig.array() * hidden2_sig_bias.array() * (1.0 - hidden2_sig_bias.array());
                dEdH2 = dEdH2bias.topRows(dEdH2bias.rows()-1);
                dEdW2 = dEdH2 * hidden1_sig_bias.transpose();
                dEdH1sig = weights2.transpose() * dEdH2;

                // 第一层梯度
                dEdH1bias = dEdH1sig.array() * hidden1_sig_bias.array() * (1.0 - hidden1_sig_bias.array());
                dEdH1 = dEdH1bias.topRows(dEdH1bias.rows()-1);
                dEdW1 = dEdH1 * x_bias.transpose();

                // 更新参数矩阵
                weights1 = weights1 - lr * dEdW1;
                weights2 = weights2 - lr * dEdW2;
                weights3 = weights3 - lr * dEdW3;
            }
            if(i % 500 == 0){
                printf("%d/%d error %f\n", i, epochs, error2);
            }
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
    MatrixXf weights3;
};


int main(){
    // 读取训练样本
    MatrixXf xt = readx("x.txt").transpose();
    MatrixXf yt = ready("y.txt").transpose();
    // 读取测试样本
    MatrixXf x_mesh = readx("x_mesh.txt").transpose();

    // nn参数
    int x_dims = xt.rows();
    int y_dims = yt.rows();
    int hidden_nums = 32;
    float lr = 1e-1;
    int epochs = 10000;
    int batch = 20;

    // nn实例化
    NN nn(x_dims, y_dims, hidden_nums, lr, epochs, batch);
    nn.train(xt, yt);
    
    // 训练样本预测
    MatrixXf p1 = nn.predict(xt);
    // 测试样本预测
    MatrixXf predict = nn.predict(x_mesh);

    // 保存测试预测结果
    savey(predict.transpose(), "y_mesh.txt");
    return 0;
}











