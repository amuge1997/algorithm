#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef _CP_
#define _CP_

void print_float(float f){
    printf("%f\n", f);
}
void print_int(int i){
    printf("%d\n", i);
}
void print_float_array(float* arr, int len){
    int i;
    for(i=0;i<len;i++){
        print_float(arr[i]);
    }
}
void print_n(){
    printf("\n");
}
void print_s(const char* s){
    printf("%s\n", s);
}
void print_int_array(int* arr, int len){
    int i;
    for(i=0;i<len;i++){
        print_int(arr[i]);
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


// 打印矩阵
void print_float_matrix(float** arr, int rows, int cols){
    for(int ri=0; ri<rows; ri++){
        for(int ci=0; ci<cols; ci++){
            printf("%f ", arr[ri][ci]);
        }
        print_n();
    }
}

// 打印矩阵
void print_int_matrix(int** arr, int rows, int cols){
    for(int ri=0; ri<rows; ri++){
        for(int ci=0; ci<cols; ci++){
            // printf("%f ", *(*(arr+ri)+ci));
            printf("%d ", arr[ri][ci]);
        }
        print_n();
    }
}

// 打印路径
void show_route(float** Q, int start_state, int end_state, int rows, int cols){
    int state = start_state;
    print_s("Best Route");
    printf("%d", state);
    while(state != end_state){
        int max_a = 0;
        float max_q = Q[state][max_a];
        for(int a=1;a<cols;a++){
            float q = Q[state][a];
            if(max_q < Q[state][a]){
                max_a = a;
                max_q = q;
            }
        }
        state = max_a;
        printf("->%d", state);
    }
}


#endif







