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


// 位置结构体
struct position
{
    int y;
    int x;
};


// 网格结构体
struct grid
{
    position pos;
    grid* from_grid;
    float f;
    float h;
    float g;

    int map_type;
};


// 距离计算,h曼哈顿距离,o欧式距离
float distance(position p1, position p2, char mode){
    float ret;
    if(mode == 'o'){
        ret = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
    }else if (mode == 'h')
    {
        ret = fabs(p1.x - p2.x) + fabs(p1.y - p2.y);
    }
    return ret;
}


// 制作网格图
void make_grid(int** the_map, grid** the_grid_map, int rows, int cols){
    for(int ri=0;ri<rows;ri++){
        for(int ci=0;ci<cols;ci++){
            position p = {ri, ci};
            grid temp = {p, NULL, -1., -1., -1., the_map[ri][ci]};
            the_grid_map[ri][ci] = temp;
        }
    }
}


// 网格列表结构体
struct grid_list{
    grid** array=NULL;
    int tail_index=-1;
};

void print_grid(grid the_grid);

// 快速排序, 根据f值对列表排序
void list_sort_(grid** arr, int len){
    grid* first = arr[0];

    int swap_index = 1;
    
    for(int i=1;i<len;i++){
        grid* now = arr[i];
        if(now->f < first->f){
            grid* temp = arr[swap_index];
            arr[swap_index] = now;
            arr[i] = temp;
            swap_index++;
        }
    }

    int mid_index = swap_index - 1;
    int j;
    for(j=0;j<mid_index;j++){
        arr[j] = arr[j+1];
    }
    arr[mid_index] = first;

    if(mid_index != 0){
        list_sort_(arr, mid_index);
    }
    if(len - swap_index != 0){
        list_sort_(arr + swap_index, len - swap_index);
    }
}

// 快速排序, 根据f值对列表排序
void list_sort_by_f(grid_list* list){
    if(list->tail_index > 0){
        list_sort_(list->array, list->tail_index+1);
    }
}

// 添加元素
void list_append(grid_list* list, grid* new_grid){
    list->tail_index++;
    list->array[list->tail_index] = new_grid;
}

// 根据索引获取元素
grid* list_get_by_index(grid_list* list, int index){
    return list->array[index];
}

// 根据索引删除元素
void list_remove_by_index(grid_list* list, int index){
    if(index <= list->tail_index && index >=0){
        for(int i=index;i<list->tail_index;i++){
            list->array[i]=list->array[i+1];
        }
        list->tail_index--;
    }
}

// 元素是否已经存在列表中
bool is_in_list(grid_list* list, grid* the_grid){
    for(int i=0;i<list->tail_index+1;i++){
        if(the_grid == list->array[i]){
            return true;
        }
    }
    return false;
}


// 打印网格
void print_grid(grid the_grid){
    print_s("y, x");
    print_int(the_grid.pos.y);
    print_int(the_grid.pos.x);
    print_s("f,h,g");
    print_float(the_grid.f);
    print_float(the_grid.h);
    print_float(the_grid.g);
    print_n();
}


// 打印网格列表
void print_grid_list(grid_list* list){
    for(int i=0; i<list->tail_index+1; i++){
        print_grid(*(list->array[i]));
    }
}

// 打印网格列表
void print_grid_list_pos(grid_list* list){
    for(int i=0; i<list->tail_index+1; i++){
        grid the_grid = *(list->array[i]);
        printf("(y,x): (%d,%d)\n", the_grid.pos.y, the_grid.pos.x);
    }
}
#endif







