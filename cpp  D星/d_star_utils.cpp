#include <stdio.h>
#include <math.h>
#include <stdlib.h>


#ifndef _DSU_
#define _DSU

#define BLOCK 1
#define INF 100000.
#define New 'n'
#define Close 'c'
#define Open 'o'


float min(float a, float b){
    if(a < b) return a;
    return b;
}

void pf(float f){
    printf("%f\n", f);
}
void pi(int i){
    printf("%d\n", i);
}
void print_float_array(float* arr, int len){
    int i;
    for(i=0;i<len;i++){
        pf(arr[i]);
    }
}
void print_n(){
    printf("\n");
}
void pn(){
    printf("\n");
}
void ps(const char* s){
    printf("%s\n", s);
}
void print_int_array(int* arr, int len){
    int i;
    for(i=0;i<len;i++){
        pi(arr[i]);
    }
}

// 位置结构体
struct position
{
    int y;
    int x;
};
void pp(position p){
    printf("y,x: %d,%d\n", p.y, p.x);
}

// 网格结构体
struct grid
{
    position pos;
    grid* parent;
    float h;
    float j;
    char state;     // n,c,o

    int map_type;
};


struct pair{
    int r;
    int c;
};

// int PANUMS = 4;
// pair PA[4] = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
int PANUMS = 8;
pair PA[8] = {
    {-1, 0}, 
    {0, -1}, 
    {0, 1}, 
    {1, 0}, 
    {1, 1}, 
    {-1, -1}, 
    {1, -1}, 
    {-1, 1}
};


// 网格列表结构体
struct grid_list{
    grid** array=NULL;
    int tail_index=-1;
};


void pg(grid* the_grid);

// 快速排序, 根据f值对列表排序
void list_sort_(grid** arr, int len){
    grid* first = arr[0];

    int swap_index = 1;
    
    for(int i=1;i<len;i++){
        grid* now = arr[i];
        if(now->j < first->j){
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
void list_sort_by_j(grid_list* list){
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
            grid temp = {p, NULL, INF, -1., 'n', the_map[ri][ci]};
            the_grid_map[ri][ci] = temp;
        }
    }
}

// 打印网格
void pg(grid* the_grid){
    printf("y, x     %d %d\n", the_grid->pos.y, the_grid->pos.x);
    
    if(the_grid->state == 'n'){
        ps("state    new");
    }
    else if(the_grid->state == 'c'){
        ps("state    close");
    }
    else if(the_grid->state == 'o'){
        ps("state    open");
    }
    if(NULL != the_grid->parent){
        printf("from     %d %d\n", the_grid->parent->pos.y, the_grid->parent->pos.x);
    }
    else{
        ps("from     NULL");
    }
    
    printf("j, h     %f %f\n", the_grid->j, the_grid->h);

    
    print_n();
}


// 打印网格列表
void print_grid_list(grid_list* list){
    for(int i=0; i<list->tail_index+1; i++){
        pg(list->array[i]);
    }
}

void pgp(grid** map, int y, int x){
    grid* g = &map[y][x];
    pg(g);
}


float cost(grid* grid1, grid* grid2){
    if(grid1->map_type == BLOCK or grid2->map_type == BLOCK){
        return INF;
    }else{
        return distance(grid1->pos, grid2->pos, 'o');
    }
}

bool a_equal_b(float a, float b){
    float epsilon = 1e-6;
    if (fabs(a - b) < epsilon) {
        return true;
    } else {
        return false;
    }
}


// 展示路径
void show_route(grid_list* lines){
    if(-1 == lines->tail_index){
        ps("no route");
    }else{
        ps("Best Route");
        for(int i=0;i<lines->tail_index;i++){
            printf("(%d,%d)->", lines->array[i]->pos.y, lines->array[i]->pos.x);
        }
        printf("(%d,%d)", lines->array[lines->tail_index]->pos.y, lines->array[lines->tail_index]->pos.x);
    }
    
}


void saveStructToTextFile(char *filename, grid** the_grid_map, int rows, int cols) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    
    for(int ri=0;ri<rows;ri++){
        for(int ci=0;ci<cols;ci++){

            grid data = the_grid_map[ri][ci];
            if(data.parent == NULL){
                fprintf(file, "%d\t%d\t%d\t%d\t%f\t%f\t%c\t%d\n", 
                data.pos.y, data.pos.x, -1, -1,
                data.j, data.h, 
                data.state, data.map_type);
            }else{
                fprintf(file, "%d\t%d\t%d\t%d\t%f\t%f\t%c\t%d\n", 
                data.pos.y, data.pos.x,
                data.parent->pos.y, data.parent->pos.x,
                data.j, data.h, 
                data.state, data.map_type);
            };
        }
    }
    
    fclose(file);
}



#endif














