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
// 写入位置
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

#endif







