#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include "./eigen-3.4.0/Eigen/Dense"


// CTRV 匀速运转速模型
#ifndef _CTRV_
#define _CTRV_

using namespace std;
using namespace Eigen;

typedef Vector<float, 5> vec5f;
typedef Matrix<float, 5, 5> mat5x5f;


// 时间步
float dt = 0.5;


// 雅可比矩阵
mat5x5f ctrv_jacobian(const vec5f& state){
    mat5x5f jacobian = mat5x5f::Zero();
    float x = state(0);
    float y = state(1);
    float v = state(2);
    float theta = state(3);
    float omega = state(4);

    jacobian(0, 0) = 1;
    jacobian(0, 2) = 1/omega * (sin(omega*dt+theta)-sin(theta));
    jacobian(0, 3) = v/omega * (cos(omega*dt+theta)-cos(theta));
    jacobian(0, 4) = v*dt/omega*cos(omega*dt+theta) - v/(omega*omega)*(sin(omega*dt+theta)-sin(theta));

    jacobian(1, 1) = 1;
    jacobian(1, 2) = 1/omega * (-cos(omega*dt+theta)+cos(theta));
    jacobian(1, 3) = v/omega * (sin(omega*dt+theta)-sin(theta));
    jacobian(1, 4) = v*dt/omega*sin(omega*dt+theta) - v/(omega*omega)*(-cos(omega*dt+theta)+cos(theta));
    
    jacobian(2, 2) = 1;

    jacobian(3, 3) = 1;
    jacobian(3, 4) = dt;

    jacobian(4, 4) = 1;

    return jacobian;
}


// 状态转移
vec5f ctrv_forward(const vec5f& state){
    vec5f next_state;
    float x = state(0);
    float y = state(1);
    float v = state(2);
    float theta = state(3);
    float omega = state(4);
    
    next_state(0) = x + v/omega * (sin(omega*dt+theta) - sin(theta));
    next_state(1) = y + v/omega * (-cos(omega*dt+theta)+ cos(theta));
    next_state(2) = v;
    next_state(3) = theta + omega*dt;
    next_state(4) = omega;
    return next_state;
}



#endif






