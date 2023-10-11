#include <stdio.h>
#include <math.h>
#include "c_print.cpp"
#define BLOCK 1


struct position
{
    /* data */
    int y;
    int x;
};


struct grid
{
    position pos;
    grid* from_grid;
    float f;
    float h;
    float g;

    int map_type;
};


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


void make_grid(int** the_map, grid** the_grid_map, int rows, int cols){
    for(int ri=0;ri<rows;ri++){
        for(int ci=0;ci<cols;ci++){
            position p = {ri, ci};
            grid temp = {p, NULL, -1., -1., -1., the_map[ri][ci]};
            the_grid_map[ri][ci] = temp;
        }
    }
}


struct grid_list{
    grid** array=NULL;
    int tail_index=-1;
};

void print_grid(grid the_grid);

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

void list_sort_by_f(grid_list* list){
    if(list->tail_index > 0){
        list_sort_(list->array, list->tail_index+1);
    }
}

void list_append(grid_list* list, grid* new_grid){
    list->tail_index++;
    list->array[list->tail_index] = new_grid;
}

grid* list_get_by_index(grid_list* list, int index){
    return list->array[index];
}

void list_remove_by_index(grid_list* list, int index){
    if(index <= list->tail_index && index >=0){
        for(int i=index;i<list->tail_index;i++){
            list->array[i]=list->array[i+1];
        }
        list->tail_index--;
    }
}

bool is_in_list(grid_list* list, grid* the_grid){
    for(int i=0;i<list->tail_index+1;i++){
        if(the_grid == list->array[i]){
            return true;
        }
    }
    return false;
}


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


void print_grid_list(grid_list* list){
    for(int i=0; i<list->tail_index+1; i++){
        print_grid(*(list->array[i]));
    }
}


void find_near_grid(
    grid** the_grid_map, 
    grid* this_grid, 
    position target_position, 
    grid_list* open_list, 
    grid_list* close_list, 
    int rows, int cols
){
    int y_low, y_up, x_low, x_up;
    y_low = 0;
    y_up = rows;
    x_low = 0;
    x_up = cols;

    for(int row=-1;row<2;row++){
        for(int col=-1;col<2;col++){
            if(row == 0 and col == 0) continue;
            position near_position = {this_grid->pos.y + row, this_grid->pos.x + col};
            if(near_position.x < x_low or near_position.x >= x_up){}
            else if (near_position.y < y_low or near_position.y >= y_up){}
            else if (the_grid_map[near_position.y][near_position.x].map_type == BLOCK){}
            else{
                grid* near_grid = &the_grid_map[near_position.y][near_position.x];
                float h = distance(target_position, near_grid->pos, 'h');
                float d = distance(this_grid->pos, near_grid->pos, 'o');
                float g = d + this_grid->g;
                if(!is_in_list(close_list, near_grid)){
                    if(!is_in_list(open_list, near_grid)){
                        float f = h + g;
                        near_grid->from_grid = this_grid;
                        near_grid->f = f;
                        near_grid->h = h;
                        near_grid->g = g;
                        list_append(open_list, near_grid);
                    }else{
                        if(near_grid->g > g){
                            near_grid->g = g;
                            near_grid->from_grid = this_grid;
                        }
                    }
                }
            }
            
        }
    }
}


void a_star(grid_list* lines, grid** the_grid_map,position start_position, position target_position, int rows, int cols, int max_epochs){
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

    grid* open_list_temp[rows * cols];
    grid_list open_list;
    open_list.array = open_list_temp;
    grid* close_list_temp[rows * cols];
    grid_list close_list;
    close_list.array = close_list_temp;
    
    list_append(&open_list, &the_grid_map[start_position.y][start_position.x]);
    
    grid* result_grid = NULL;
    for(int i=0;i<max_epochs;i++){
        grid* this_grid = list_get_by_index(&open_list, 0);
        list_remove_by_index(&open_list, 0);
        list_append(&close_list, this_grid);
        find_near_grid(the_grid_map, this_grid, target_position, &open_list, &close_list, rows, cols);
        list_sort_by_f(&open_list);
        
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


void show_route(grid_list* lines){
    print_s("Best Route");
    int i=lines->tail_index;
    printf("(%d,%d)", lines->array[i]->pos.y, lines->array[i]->pos.x);
    for(i=lines->tail_index-1;i>=0;i--){
        printf("->(%d,%d)", lines->array[i]->pos.y, lines->array[i]->pos.x);
    }
}


void run(){
    // int rows = 7;
    // int cols = 7;
    // int the_map_temp[rows][cols] = {
    //     {0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 1, 1, 1, 0, 0},
    //     {0, 0, 0, 0, 1, 0, 0},
    //     {0, 0, 1, 1, 1, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0},
    // };
    // position start_position = {0, 0};
    // position target_position = {6, 6};

    int rows = 7;
    int cols = 7;
    int the_map_temp[rows][cols] = {
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 0, 0},
        {0, 0, 0, 0, 1, 0, 0},
        {0, 0, 1, 1, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
    };
    position start_position = {3, 1};
    position target_position = {3, 5};


    int* the_map[rows];
    for(int ri=0;ri<rows;ri++){
        the_map[ri] = the_map_temp[ri];
    }

    grid the_grid_map_temp[rows][cols];
    grid* the_grid_map[rows];
    for(int ri=0;ri<rows;ri++){
        the_grid_map[ri] = the_grid_map_temp[ri];
    }
    
    make_grid(the_map, the_grid_map, rows, cols);

    grid* lines_list_temp[1000];
    grid_list lines;
    lines.array = lines_list_temp;
    a_star(&lines, the_grid_map, start_position, target_position, rows, cols, 20);

    print_n();
    // print_grid_list(&lines);
    show_route(&lines);
}


int main(){
    
    run();
    return 0;
}












