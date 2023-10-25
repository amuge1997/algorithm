#include <stdio.h>
#include <time.h>
#include "sa_utils.cpp"


// 初始解、中间解、最优解
struct solution_trip{
    array_int* start;
    array_int* middle;
    array_int* best;
};
// 距离计算
float distance(position p1, position p2){
    float ret = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
    return ret;
}
// 计算距离矩阵
void cal_dis_matrix(position* pos_set, int len, float** dis_mat){
    for(int ri=0;ri<len;ri++){
        for(int ci=0;ci<len;ci++){
            dis_mat[ri][ci] = distance(pos_set[ri], pos_set[ci]);
        }
    }
}
// 不放回采样两个索引
void sample_tow_index(int len, int* result1, int* result2){
    int arr_[len];
    array_int arr;
    arr.array = arr_;
    for(int i=0;i<len;i++){
        array_int_append(&arr, i);
    }
    int index = rand_int(0, arr.tail_index+1);
    *result1 = index;
    array_remove_by_index(&arr, index);
    index = rand_int(0, arr.tail_index+1);
    *result2 = index;
}
// 初始解
void init_solation(array_int* solution){
    int len = solution->tail_index+1;
    int arr_[len];
    array_int arr;
    arr.array = arr_;
    for(int i=0;i<len;i++){
        array_int_append(&arr, i);
    }
    
    for(int i=0;i<len;i++){
        int index = rand_int(0, arr.tail_index+1);
        solution->array[i] = arr.array[index];
        array_remove_by_index(&arr, index);
    }

}
// 计算解的总距离
float total_distance(float** dis_mat, array_int* solution){
    float total_dis=0.;
    int len = solution->tail_index+1;
    int i1, i2;
    for(int i=0;i<len-1;i++){
        i1=solution->array[i];
        i2=solution->array[i+1];
        total_dis += dis_mat[i1][i2];
    }
    i1=solution->array[len-1];
    i2=solution->array[0];
    total_dis += dis_mat[i1][i2];
    return total_dis;
}
// 模拟退火
void sa(float** dis_mat, float init_temp, float rate, int len, int epochs, solution_trip* solution3, float* best_dist){

    array_int* start_solution = solution3->start;
    array_int* middle_solution = solution3->middle;
    array_int* best_solution = solution3->best;

    array_int curr_solution;
    int arr_[len];
    curr_solution = {arr_, len-1};

    init_solation(&curr_solution);
    float curr_dist = total_distance(dis_mat, &curr_solution);

    array_int_copy(&curr_solution, best_solution);
    *best_dist = curr_dist;

    array_int new_solution;
    int ns_[len];
    new_solution = {ns_, len-1};

    int r1, r2;
    float temp, new_dist, mid;
    // 保存初始解
    array_int_copy(&curr_solution, start_solution);
    for(int ep=0;ep<epochs;ep++){

        // 随机选取位置进行调换
        array_int_copy(&curr_solution, &new_solution);
        sample_tow_index(len, &r1, &r2);
        mid = new_solution.array[r1];
        new_solution.array[r1] = new_solution.array[r2];
        new_solution.array[r2] = mid;

        // 计算总距离
        new_dist = total_distance(dis_mat, &new_solution);

        // 退火
        temp = init_temp * pow(rate, ep);
        // 是否选择解
        if(new_dist < curr_dist || rand_float(0, exp(- (new_dist - curr_dist) / temp))){
            array_int_copy(&new_solution, &curr_solution);
            curr_dist = new_dist;
        }
        // 保存中间解
        if(epochs / 2 == ep){
            array_int_copy(&curr_solution, middle_solution);
        }
        // 保存最优解
        if(new_dist < *best_dist){
            array_int_copy(&new_solution, best_solution);
            *best_dist = new_dist;
        }
        
    }

}


int main(){
    // // 读取坐标
    int len = 10;
    position pos_set[len];
    srand(128);
    rand_position_generate(pos_set, len);
    write_position("position.txt", pos_set, len);
    read_position("position.txt", pos_set, len);

    // 计算距离矩阵
    float dis_mat_[len][len];
    float* dis_mat[len];
    for(int li=0;li<len;li++){
        dis_mat[li] = dis_mat_[li];
    }
    cal_dis_matrix(pos_set, len, dis_mat);

    // 算法参数
    float init_temp = 100.0;
    float rate = 0.98;
    int epochs = 1000;

    // 保存初始解、中间解、最优解
    solution_trip solution3;

    array_int start_solution;
    int bs2[len];
    start_solution = {bs2, len-1};

    array_int middle_solution;
    int bs3[len];
    middle_solution = {bs3, len-1};
    
    array_int best_solution;
    int bs1[len];
    best_solution = {bs1, len-1};
    float best_dist;

    solution3.start = &start_solution;
    solution3.middle = &middle_solution;
    solution3.best = &best_solution;

    // 模拟退火
    sa(dis_mat, init_temp, rate, len, epochs, &solution3, &best_dist);

    // 保存结果
    save_solution("solution_start.txt", start_solution.array, len);
    save_solution("solution_middle.txt", middle_solution.array, len);
    save_solution("solution_best.txt", best_solution.array, len);

    return 0;
}











