#include <stdio.h>
#include <stdlib.h>

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
#endif







