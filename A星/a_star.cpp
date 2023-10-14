#include <stdio.h>
#include <math.h>
#include "c_print.cpp"
#define BLOCK 1


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


struct pair{
    int r;
    int c;
};


// 探索附近网格
void find_near_grid(
    grid** the_grid_map, 
    grid* this_grid, 
    position target_position, 
    grid_list* open_list, 
    grid_list* close_list, 
    int rows, int cols
){
    // 边界
    int y_low, y_up, x_low, x_up;
    y_low = 0;
    y_up = rows;
    x_low = 0;
    x_up = cols;

    pair pa[4] = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};

    for(int i=0;i<4;i++){
        int row = pa[i].r;
        int col = pa[i].c;

        position near_position = {this_grid->pos.y + row, this_grid->pos.x + col};
        if(near_position.x < x_low or near_position.x >= x_up){}
        else if (near_position.y < y_low or near_position.y >= y_up){}
        else if (the_grid_map[near_position.y][near_position.x].map_type == BLOCK){}
        else{
            // 计算附近网格g和h
            grid* near_grid = &the_grid_map[near_position.y][near_position.x];
            float h = distance(target_position, near_grid->pos, 'h');
            float d = distance(this_grid->pos, near_grid->pos, 'o');
            float g = d + this_grid->g;
            if(!is_in_list(close_list, near_grid)){
                if(!is_in_list(open_list, near_grid)){
                    // 未出现在待探索列表中
                    float f = h + g;
                    near_grid->from_grid = this_grid;
                    near_grid->f = f;
                    near_grid->h = h;
                    near_grid->g = g;
                    list_append(open_list, near_grid);
                }else{
                    // 已出现在待探索列表中
                    if(near_grid->g > g){
                        // 若从当前网格到附近网格距离更优,更新附近网格g值
                        near_grid->g = g;
                        near_grid->from_grid = this_grid;
                    }
                }
            }
        }
    }
    
}


// A星
void a_star(
    grid** the_grid_map,
    position start_position, 
    position target_position, 
    int rows, int cols,
    grid_list* lines
    ){

    // 初始格
    float h = distance(start_position, target_position, 'h');
    float g = distance(start_position, start_position, 'o');
    float f = h + g;
    grid start_grid = {
        start_position, 
        NULL, 
        f, h, g, 
        the_grid_map[start_position.y][start_position.x].map_type
    };

    the_grid_map[start_position.y][start_position.x] = start_grid;

    // 待搜索列表
    grid* open_list_temp[rows * cols];
    grid_list open_list;
    open_list.array = open_list_temp;
    // 已搜索列表
    grid* close_list_temp[rows * cols];
    grid_list close_list;
    close_list.array = close_list_temp;
    
    list_append(&open_list, &the_grid_map[start_position.y][start_position.x]);
    
    // 开始执行
    grid* result_grid = NULL;
    while(-1 != open_list.tail_index){
        // 取出f最小值的网格
        grid* this_grid = list_get_by_index(&open_list, 0);
        list_remove_by_index(&open_list, 0);
        list_append(&close_list, this_grid);

        // 探索周围网格
        find_near_grid(the_grid_map, this_grid, target_position, &open_list, &close_list, rows, cols);
        list_sort_by_f(&open_list);

        // 判断目标网格是否出现
        for(int j=0;j<open_list.tail_index+1;j++){
            grid* temp = open_list.array[j];
            if(temp->pos.x == target_position.x && temp->pos.y == target_position.y){
                result_grid = temp;
                break;
            }
        }
        if(NULL != result_grid) break;
    }
    
    if(result_grid != NULL){
        grid* this_grid = result_grid;
        while (1){
            list_append(lines, this_grid);
            this_grid = this_grid->from_grid;
            if(this_grid->from_grid == NULL){
                list_append(lines, this_grid);
                break;
            }
        }
    }
    
}


// 展示路径
void show_route(grid_list* lines){
    if(-1 == lines->tail_index){
        print_s("no route");
    }else{
        print_s("Best Route");
        int i=lines->tail_index;
        printf("(%d,%d)", lines->array[i]->pos.y, lines->array[i]->pos.x);
        for(i=lines->tail_index-1;i>=0;i--){
            printf("->(%d,%d)", lines->array[i]->pos.y, lines->array[i]->pos.x);
        }
    }
    
}


// 绘制
void draw_result(
    int** the_map,
    position start_position, 
    position target_position,
    grid_list* lines,
    int rows, int cols
){
    for(int ri=0;ri<rows;ri++){
        for(int ci=0;ci<cols;ci++){
            
            bool is_draw = false;
            if(ri==start_position.y && ci==start_position.x){
                printf("G ");
                is_draw = true;
            }
            else if (ri==target_position.y && ci==target_position.x)
            {
                printf("X ");
                is_draw = true;
            }

            for(int i=0;i<lines->tail_index+1;i++){
                if(ri == lines->array[i]->pos.y && ci == lines->array[i]->pos.x && !is_draw){
                    // printf("● ");
                    printf("○ ");
                    is_draw = true;
                    break;
                }
            }
            if(!is_draw){
                if(BLOCK == the_map[ri][ci]){
                    printf("■ ");
                    // printf("□ ");
                }
                else{
                    printf("  ");
                }
            }

        }
        print_n();
    }
    print_n();
    print_n();

}


void run(){
    // 地图构建
    int rows = 9;
    int cols = 15;
    int the_map_temp[rows][cols] = {
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1},
        {1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1},
        {1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    };
    position start_position = {4, 2};
    position target_position = {3, 12};


    int* the_map[rows];
    for(int ri=0;ri<rows;ri++){
        the_map[ri] = the_map_temp[ri];
    }

    // 制作网格
    grid the_grid_map_temp[rows][cols];
    grid* the_grid_map[rows];
    for(int ri=0;ri<rows;ri++){
        the_grid_map[ri] = the_grid_map_temp[ri];
    }
    make_grid(the_map, the_grid_map, rows, cols);

    // 算法执行
    grid* lines_list_temp[1000];
    grid_list lines;
    lines.array = lines_list_temp;
    a_star(the_grid_map, start_position, target_position, rows, cols, &lines);
    
    // 展示路径
    printf("Start: (%d,%d)\n  End: (%d,%d)\n", start_position.y, start_position.x, target_position.y, target_position.x);
    // print_n();
    show_route(&lines);
    
    printf("\n\n\n");

    draw_result(the_map, start_position, target_position, &lines, rows, cols);
}


int main(){
    run();
    return 0;
}












