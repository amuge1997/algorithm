#include <stdio.h>
#include "q_star_utils.cpp"


// Q学习
void q_learn(int** adj_matrix, float** returns_matrix, float** Q, int start_states, int end_states, int rows, int cols, int epochs){

    // 选择随机动作的概率
    float p = 0.5;

    float y = 0.1;
    float w = 0.9;

    // 迭代
    for(int i=0;i<epochs;i++){
        int state = start_states;
        while(state != end_states){
            int select_a;
            // 选择动作
            float e = rand_float(0, 1);
            array_int available_actions;
            int available_actions_[cols];
            available_actions.array=available_actions_;
            for(int a=0;a<cols;a++){
                if(adj_matrix[state][a] == 1){
                    array_int_append(&available_actions, a);
                }
            }
            if(e < p){
                // 随机选择动作a
                int rand_a_index = rand_int(0, available_actions.tail_index+1);
                select_a = available_actions.array[rand_a_index];
            }else{
                // 根据最大Q值选择动作
                int select_temp_a = available_actions.array[0];
                float select_temp_a_Q = Q[state][select_temp_a];
                for(int t=1;t<available_actions.tail_index+1;t++){
                    float q = Q[state][available_actions.array[t]];
                    if(select_temp_a_Q < q){
                        select_temp_a = available_actions.array[t];
                        select_temp_a_Q = q;
                    }
                }
                select_a = select_temp_a;
            }
            
            // 更新Q表
            float this_s_ret = returns_matrix[state][select_a];
            int next_s = select_a;
            float* next_s_Q = Q[next_s];
            float next_s_max_Q = next_s_Q[0];
            for(int k=1;k<cols;k++){
                float q = next_s_Q[k];
                if(next_s_max_Q < q) next_s_max_Q=q;
            }
            Q[state][select_a] = Q[state][select_a] + w * (this_s_ret + y * (next_s_max_Q - Q[state][select_a]));
            // 进入下一个状态
            state = select_a;
        }
    }
}


void run(){
    
    // /*
    // map
    //     0---1---
    //         |   |
    //     2---3---5
    //         |   |
    //         4---
    // */
    // 地图构建
    // int rows = 6;
    // int cols = 6;
    // // 邻接矩阵
    // int adj_matrix_[rows][cols] = {
    //     {0, 1, 0, 0, 0, 0},
    //     {1, 0, 0, 1, 0, 1},
    //     {0, 0, 0, 1, 0, 0},
    //     {0, 1, 1, 0, 1, 0},
    //     {0, 0, 0, 1, 0, 1},
    //     {0, 1, 0, 0, 1, 0},
    // };
    // // 奖励矩阵
    // float returns_matrix_[rows][cols] = {
    //     {0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 100},
    //     {0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 100},
    //     {0, 0, 0, 0, 0, 0},
    // };
    // // 起点与终点
    // int start_state = 2;
    // int end_state = 5;

    /*
    map
        0---1---
            |   |
        2---3---5
            |   |
            4---6---7
    */
    // 地图构建
    int rows = 8;
    int cols = 8;
    // 邻接矩阵
    int adj_matrix_[rows][cols] = {
        {0, 1, 0, 0, 0, 0, 0, 0},
        {1, 0, 0, 1, 0, 1, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 1, 1, 0, 1, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 1, 0},
        {0, 1, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 1, 1, 0, 1},
        {0, 0, 0, 0, 0, 0, 1, 0},
    };
    // 奖励矩阵
    float returns_matrix_[rows][cols] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 100},
        {0, 0, 0, 0, 0, 0, 0, 0},
    };
    // 起点与终点
    int start_state = 2;
    int end_state = 7;
    

    int* adj_matrix[rows];
    for(int ri=0;ri<rows;ri++){
        adj_matrix[ri] = adj_matrix_[ri];
    }
    float* returns_matrix[rows];
    for(int ri=0;ri<rows;ri++){
        returns_matrix[ri] = returns_matrix_[ri];
    }

    // Q表
    float Q_[rows][cols]={
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
    };
    float* Q[rows];
    for(int ri=0;ri<rows;ri++){
        Q[ri] = Q_[ri];
    }

    // 算法执行
    q_learn(adj_matrix, returns_matrix, Q, start_state, end_state, rows, cols, 10);

    // 打印Q表
    print_s("Q table");
    print_float_matrix(Q, rows, cols);
    print_n();
    // 打印路径
    printf("Start: %d\n  End: %d\n", start_state, end_state);
    print_n();
    show_route(Q, start_state, end_state, rows, cols);
    print_n();
}


int main(){
    run();
    return 0;
}










