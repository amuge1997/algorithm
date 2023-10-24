#include <stdio.h>
#include <math.h>
#include <time.h>
#include "c_print.cpp"


// 随机int
int rand_int(int start, int end){
    int ret = rand()%(end-start) + start;
    return ret;
}

// 随机float
float rand_float(float start, float end){
    float s = 1.*rand() / RAND_MAX;
    float ret = (end - start) * s + start;
    return ret;
};

// 位置结构体
struct position
{
    float y;
    float x;
};

// 读取位置
void read_position(const char* filename, position* pos_set, int len) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open the file: %s\n", filename);
        return;
    }

    for (int i = 0; i < len; i++) {
        float y, x;
        int result = fscanf(file, "%f\t%f", &y, &x);
        if (result != 2) {
            printf("Failed to read file.\n");
            break;
        }
        pos_set[i].y = y;
        pos_set[i].x = x;
        
    }
    fclose(file);
}
void write_position(const char *filename, position* pos_set, int len) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Failed to open the file: %s\n", filename);
        return;
    }

    for (int i = 0; i < len; i++) {
        fprintf(file, "%f\t%f\n", pos_set[i].y, pos_set[i].x);
    }

    fclose(file);
}

// 保存解
void save_solution(const char *filename, const int* arr, int len) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Failed to open the file: %s\n", filename);
        return;
    }
    for (int i = 0; i < len; i++) {
        fprintf(file, "%d\n", arr[i]);
    }
    fclose(file);
}

// 生成随机坐标
void rand_position_generate(position* pos_set, int len){
    for(int i=0;i<len;i++){
        for(int j=0;j<2;j++){
            pos_set[i].y = rand_float(-2, 2);
            pos_set[i].x = rand_float(-2, 2);
        }
    }
}


// 数组结构体
struct array_int
{
    int* array;
    int tail_index=-1;
};
// 数据添加元素
void array_int_append(array_int* array, int value){
    array->array[array->tail_index+1] = value;
    array->tail_index++;
}
// 根据索引获取元素
int array_int_get_by_index(array_int* array, int index){
    return array->array[index];
}
// 根据索引删除元素
void array_remove_by_index(array_int* array, int index){
    if(index <= array->tail_index && index >=0){
        for(int i=index;i<array->tail_index;i++){
            array->array[i]=array->array[i+1];
        }
        array->tail_index--;
    }
}
// 复制
void array_int_copy(array_int* from_array, array_int* to_array){
    int len = from_array->tail_index+1;
    for(int i=0;i<len;i++){
        to_array->array[i] = from_array->array[i];
    }
}
// 打印结构体数组
void print_struct_array_int(array_int* array){
    for(int i=0;i<array->tail_index+1;i++){
        printf("%d ", array->array[i]);
    }
    print_n();
}


void print_array_int(int* array, int len){
    for(int i=0;i<len;i++){
        printf("%d ", array[i]);
    }
}


// 打印矩阵
void print_float_matrix(float** arr, int rows, int cols){
    for(int ri=0; ri<rows; ri++){
        for(int ci=0; ci<cols; ci++){
            printf("%f ", arr[ri][ci]);
        }
        print_n();
    }
}


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
    array_int_copy(&curr_solution, start_solution);
    for(int ep=0;ep<epochs;ep++){

        // 随机选取位置进行调换
        array_int_copy(&curr_solution, &new_solution);
        sample_tow_index(len, &r1, &r2);
        mid = new_solution.array[r1];
        new_solution.array[r1] = new_solution.array[r2];
        new_solution.array[r2] = mid;

        new_dist = total_distance(dis_mat, &new_solution);

        // 退火
        temp = init_temp * pow(rate, ep);
        // 是否选择解
        if(new_dist < curr_dist || rand_float(0, exp(- (new_dist - curr_dist) / temp))){
            array_int_copy(&new_solution, &curr_solution);
            curr_dist = new_dist;
        }
        if(new_dist < *best_dist){
            array_int_copy(&new_solution, best_solution);
            *best_dist = new_dist;
        }

        if(epochs / 2 == ep){
            array_int_copy(&curr_solution, middle_solution);
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

    save_solution("solution_start.txt", start_solution.array, len);
    save_solution("solution_middle.txt", middle_solution.array, len);
    save_solution("solution_best.txt", best_solution.array, len);

    return 0;
}











