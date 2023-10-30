#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "d_star_utils.cpp"


void insert(grid_list* openlist, grid* X, float h_new){
    if(X->state == New){
        X->j = h_new;
    }
    if(X->state == Open){
        X->j = min(X->j, h_new);
    }
    if(X->state == Close){
        X->j = min(X->h, h_new);
    }
    X->state = Open;
    X->h = h_new;

    if(!is_in_list(openlist, X)) list_append(openlist, X);
    list_sort_by_j(openlist);
}

void detect_obs(grid** the_grid_map, int** new_map, grid_list* obslist, int rows, int cols){
    for(int ri=0;ri<rows;ri++){
        for(int ci=0;ci<cols;ci++){
            if(the_grid_map[ri][ci].map_type != new_map[ri][ci])
            {
                the_grid_map[ri][ci].map_type = BLOCK;
                list_append(obslist, &the_grid_map[ri][ci]);
            }
        }
    }
}

void insert_obs_and_near(grid** map, grid_list* openlist, grid_list* obslist, int rows, int cols){
    int y_low, y_up, x_low, x_up;
    y_low = 0;
    y_up = rows;
    x_low = 0;
    x_up = cols;
    for(int i=0;i<obslist->tail_index+1;i++){
        grid* obs = obslist->array[i];
        insert(openlist, obs, obs->h);
        for(int i=0;i<PANUMS;i++){
            int row = PA[i].r;
            int col = PA[i].c;
            position Y_position = {obs->pos.y + row, obs->pos.x + col};
            if(Y_position.x < x_low || Y_position.x >= x_up){}
            else if (Y_position.y < y_low || Y_position.y >= y_up){}
            else if (map[Y_position.y][Y_position.x].map_type == BLOCK){}
            else{
                grid* near = &map[Y_position.y][Y_position.x];
                insert(openlist, near, near->h);
            }
        }
    }
}

grid* get_min_j_grid(grid_list* openlist){
    if(openlist->tail_index == -1){
        return NULL;
    }
    grid* g = list_get_by_index(openlist, 0);
    return g;
}

float get_min_j(grid_list* openlist){
    if(openlist->tail_index == -1){
        return -1;
    }
    else{
        grid* g = list_get_by_index(openlist, 0);
        return g->j;
    }
}

float process_state(grid_list* openlist, grid** the_grid_map, int rows, int cols){
    // 边界
    int y_low, y_up, x_low, x_up;
    y_low = 0;
    y_up = rows;
    x_low = 0;
    x_up = cols;
    
    grid* X = get_min_j_grid(openlist);
    if(NULL == X) return -1;
    list_remove_by_index(openlist, 0);
    X->state = Close;
    (openlist);

    float j_old = X->j;

    if(j_old < X->h){
        for(int i=0;i<PANUMS;i++){
            int row = PA[i].r;
            int col = PA[i].c;
            position Y_position = {X->pos.y + row, X->pos.x + col};
            if(Y_position.x < x_low || Y_position.x >= x_up){}
            else if (Y_position.y < y_low || Y_position.y >= y_up){}
            else if (the_grid_map[Y_position.y][Y_position.x].map_type == BLOCK){}
            else{
                grid* Y = &the_grid_map[Y_position.y][Y_position.x];
                if(Y->h < j_old && X->h > Y->h + cost(Y, X)){
                    X->parent = Y;
                    X->h = Y->h + cost(Y, X);
                }
            }
        }
    }

    if(a_equal_b(j_old, X->h)){
        for(int i=0;i<PANUMS;i++){
            int row = PA[i].r;
            int col = PA[i].c;
            position Y_position = {X->pos.y + row, X->pos.x + col};
            if(Y_position.x < x_low || Y_position.x >= x_up){}
            else if (Y_position.y < y_low || Y_position.y >= y_up){}
            else{
                grid* Y = &the_grid_map[Y_position.y][Y_position.x];
                if(
                    Y->state == New ||
                    Y->parent == X && !a_equal_b(Y->h, X->h + cost(X, Y))  ||
                    Y->parent != X && Y->h > X->h + cost(X, Y)
                ){
                    Y->parent = X;
                    insert(openlist, Y, X->h + cost(X, Y));
                }
            }
        }
    }else{
        for(int i=0;i<PANUMS;i++){
            int row = PA[i].r;
            int col = PA[i].c;
            position Y_position = {X->pos.y + row, X->pos.x + col};
            if(Y_position.x < x_low || Y_position.x >= x_up){}
            else if (Y_position.y < y_low || Y_position.y >= y_up){}
            else{
                grid* Y = &the_grid_map[Y_position.y][Y_position.x];
                if(
                    Y->state == New or 
                    Y->parent == X and !a_equal_b(Y->h, X->h + cost(X, Y))
                ){
                    Y->parent = X;
                    insert(openlist, Y, X->h + cost(X, Y));
                }else{
                    if(
                        Y->parent != X and Y->h > X->h + cost(X, Y)
                    ){
                        insert(openlist, X, X->h);
                    }else if(
                        Y->parent != X and X->h > Y->h + cost(Y,X) and Y->state==Close and Y->h>j_old
                    ){
                        insert(openlist, Y, Y->h);
                    }
                }
            }
        }
    }

    list_sort_by_j(openlist);
    float min_j = get_min_j(openlist);
    return min_j;
    
}


void d_star(
    grid** the_grid_map,
    int** new_map,
    position start_position, 
    position target_position, 
    int rows, int cols,
    grid_list* lines
){
    grid_list openlist;
    grid* arr1[rows * cols];
    openlist.array = arr1;

    grid* start_grid = &the_grid_map[start_position.y][start_position.x];
    grid* traget_grid = &the_grid_map[target_position.y][target_position.x];
    traget_grid->h = 0;
    traget_grid->j = 0;

    list_append(&openlist, traget_grid);

    float min_j = process_state(&openlist, the_grid_map, rows, cols);

    int file_index=0;
    char filename[20];
    while(start_grid->state != Close && openlist.tail_index != -1)
    {
        sprintf(filename, "./grid/grid_%d.txt", file_index);
        saveStructToTextFile(filename, the_grid_map, rows, cols);
        file_index++;
        process_state(&openlist, the_grid_map, rows, cols);
    }

    if(start_grid->parent != NULL){
        grid* this_grid = start_grid;
        while (1){
            list_append(lines, this_grid);
            this_grid = this_grid->parent;
            if(this_grid->parent == NULL){
                list_append(lines, this_grid);
                break;
            }
        }
    }
    printf("遍历完成  %d %d %d\n", file_index, start_grid->state == Close, openlist.tail_index == -1);
    sprintf(filename, "./grid/grid_%d.txt", file_index);
    saveStructToTextFile(filename, the_grid_map, rows, cols);
    file_index++;

    grid* location = start_grid;
    while(true)
    {   
        grid_list obslist;
        grid* arr2[rows * cols];
        obslist.array = arr2;

        detect_obs(the_grid_map, new_map, &obslist, rows, cols);

        insert_obs_and_near(the_grid_map, &openlist, &obslist, rows, cols);

        process_state(&openlist, the_grid_map, rows, cols);
        
        sprintf(filename, "./grid/grid_%d.txt", file_index);
        saveStructToTextFile(filename, the_grid_map, rows, cols);
        if(openlist.tail_index == -1 or openlist.array[0]->j > location->h){
            printf("退出 %d %d %d", file_index+1, openlist.tail_index == -1, openlist.array[0]->j > location->h);
            break;
        }
        file_index++;
    }
    saveStructToTextFile("./grid/grid_replan_end.txt", the_grid_map, rows, cols);
}


void run(){
    // 地图构建
    int rows = 10;
    int cols = 10;
    position start_position = {4, 4};
    position target_position = {9, 9};

    int the_map_[rows][cols] = {
        {0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };
    int new_map_[rows][cols] = {
        {0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };


    int* the_map[rows];
    for(int ri=0;ri<rows;ri++){
        the_map[ri] = the_map_[ri];
    }
    int* new_map[rows];
    for(int ri=0;ri<rows;ri++) new_map[ri] = new_map_[ri];

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

    // D星
    d_star(the_grid_map, new_map, start_position, target_position, rows, cols, &lines);
}


int main(){
    run();
    return 0;
}







