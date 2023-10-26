#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <ctime>
#include "./eigen-3.4.0/Eigen/Dense"


#ifndef _NNU_
#define _NNU_


using namespace std;
using namespace Eigen;


template<typename T>
void print(const T& t){
    cout<< t << endl;
}

// 读取输入样本数据
MatrixXf readx(const string& filename) {
    ifstream input_file(filename);      // 打开文件
    MatrixXf x;                         // 存储坐标的 Eigen 矩阵
    if (input_file.is_open()) {
        vector<vector<float>> data;
        string line;
        while (getline(input_file, line)) {
            istringstream ss(line);
            vector<float> row;
            float value;
            while (ss >> value) {
                row.push_back(value);
                if (ss.peek() == '\t')
                    ss.ignore();
            }
            data.push_back(row);
        }
        input_file.close();
        // 将数据从二维向量转换为 Eigen 矩阵
        if (!data.empty()) {
            int rows = data.size();
            int cols = data[0].size();
            x.resize(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    x(i, j) = data[i][j];
                }
            }
        }
    } else {
        cerr << "Unable to open file " << filename << endl;
        // 返回一个空矩阵作为错误标志
        return MatrixXf();
    }
    return x;
}

// 读取标签样本数据
MatrixXf ready(const string& filename) {
    ifstream input_file(filename);  // 打开文件
    MatrixXf y;                     // 存储坐标的 Eigen 矩阵
    if (input_file.is_open()) {
        vector<vector<float>> data;
        string line;
        while (getline(input_file, line)) {
            istringstream ss(line);
            vector<float> row;
            float value;
            ss >> value;
            row.push_back(value);
            data.push_back(row);
        }
        input_file.close();
        // 将数据从一维向量转换为 Eigen 矩阵
        if (!data.empty()) {
            int rows = data.size();
            int cols = data[0].size();
            y.resize(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    y(i, j) = data[i][j];
                }
            }
        }
    } else {
        cerr << "Unable to open file " << filename << endl;
        // 返回一个空矩阵作为错误标志
        return MatrixXf();
    }
    return y;
}


void savey(const MatrixXf& matrix, const string& filename) {
    ofstream output_file(filename);
    if (output_file.is_open()) {
        for (int i = 0; i < matrix.rows(); i++) {
            output_file << matrix(i, 0) << '\n';
        }
        
        output_file.close();
        cout << "Matrix saved to " << filename << "." << endl;
    } else {
        cerr << "Unable to open file for writing." << endl;
    }
}


#endif










