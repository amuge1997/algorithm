#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include "./eigen-3.4.0/Eigen/Dense"
#include "nn_utils.cpp"


using namespace std;
using namespace Eigen;


class NN{
    public:

    NN(int x_dims, int y_dims, int hidden_dims, float lr, int epochs, int batch){
        this->x_dims = x_dims;
        this->y_dims = y_dims;
        this->hidden_dims = hidden_dims;
        this->lr = lr;
        this->epochs = epochs;
        this->batch = batch;
        this->weights1.resize(hidden_dims, x_dims+1);
        this->weights2.resize(y_dims, hidden_dims+1);
    }

    MatrixXf forward(const MatrixXf& x){
        int nums = x.cols();
        int x_dims = this->x_dims;
        int y_dims = this->y_dims;
        int hidden_dims = this->hidden_dims;
        
        MatrixXf weights1 = this->weights1;
        MatrixXf weights2 = this->weights2;
        MatrixXf oo(1, nums);
        MatrixXf x_one(x_dims + 1, nums);
        MatrixXf hidden(hidden_dims, nums);
        MatrixXf hidden_sig(hidden_dims, nums);
        MatrixXf hidden_sig_one(hidden_dims+1, nums);
        MatrixXf out(y_dims, nums);
        
        oo.setOnes();
        x_one << x, oo;

        hidden = weights1 * x_one;

        hidden_sig = 1.0 / (1.0 + (-hidden.array()).exp());
        hidden_sig_one << hidden_sig, oo;

        out = weights2 * hidden_sig_one;
        return out;
    }

    void train(const MatrixXf& x_ori, const MatrixXf& y_ori){
        float lr = this->lr;
        int epochs = this->epochs;
        
        int nums = x_ori.cols();
        int batch = this->batch;
        int x_dims = this->x_dims;
        int hidden_dims = this->hidden_dims;
        int y_dims = this->y_dims;

        MatrixXf weights1(hidden_dims, x_dims+1);
        MatrixXf weights2(y_dims, hidden_dims+1);
        MatrixXf oo(1, batch);
        MatrixXf x(x_dims, batch);
        MatrixXf x_one(x_dims + 1, batch);
        MatrixXf hidden(hidden_dims, batch);
        MatrixXf hidden_sig(hidden_dims, batch);
        MatrixXf hidden_sig_one(hidden_dims+1, batch);
        MatrixXf y(y_dims, batch);
        MatrixXf out(y_dims, batch);
        MatrixXf error(y_dims, batch);
        float error2;
        MatrixXf dEdW2(y_dims, hidden_dims+1);
        MatrixXf dEdHsig(hidden_dims+1, batch);
        MatrixXf dEdHone(hidden_dims+1, batch);
        MatrixXf dEdH(hidden_dims, batch);
        MatrixXf dEdW1(hidden_dims, x_dims+1);
        
        weights1 = MatrixXf::Random(hidden_dims, x_dims+1).array() / sqrt(hidden_dims + x_dims+1);
        weights2 = MatrixXf::Random(y_dims, hidden_dims+1).array() / sqrt(y_dims + hidden_dims+1);
        oo.setOnes();
        
        std::srand(unsigned(std::time(nullptr)));
        vector<int> numbers;
        for (int i = 0; i < nums; i++) {
            numbers.push_back(i);
        }
        for(int i=0;i<epochs;i++){

            random_shuffle(numbers.begin(), numbers.end());

            for (int i = 0; i < batch; i++) {
                x.col(i) = x_ori.col(numbers[i]);
                y.col(i) = y_ori.col(numbers[i]);
            }
            
            x_one << x, oo;
            
            hidden = weights1 * x_one;
            hidden_sig = 1.0 / (1.0 + (-hidden.array()).exp());
            hidden_sig_one << hidden_sig, oo;
            out = weights2 * hidden_sig_one;

            error = y - out;
            error2 = error.array().square().sum();
            
            dEdW2 = -2. * (error * hidden_sig_one.transpose()).array();
            dEdHsig = -2. * (weights2.transpose() * error).array();
            dEdHone = hidden_sig_one.array() * (1.0 - hidden_sig_one.array());
            dEdH = dEdHone.topRows(dEdHone.rows()-1);
            dEdW1 = dEdH * x_one.transpose();
            
            weights1 = weights1 - lr * dEdW1;
            weights2 = weights2 - lr * dEdW2;

        }
        this->weights1 = weights1;
        this->weights2 = weights2;
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
    
    MatrixXf p1 = nn.forward(xt);
    MatrixXf predict = nn.forward(x_mesh);

    savey(predict.transpose(), "y_mesh.txt");

    return 0;
}











