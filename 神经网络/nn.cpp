#include <stdio.h>
#include <math.h>


struct matrix{
    float** mat;
    int rows;
    int cols;
};

// 打印矩阵
void print_matrix(const matrix& mat){
    for(int ri=0; ri<mat.rows; ri++){
        for(int ci=0; ci<mat.cols; ci++){
            printf("%f ", mat.mat[ri][ci]);
        }
        printf("\n");
    }
    printf("\n");
}

void dot(const matrix& left, const matrix& righ, const matrix& result){
    // a*b b*c
    int m = left.rows;
    int n = left.cols;
    int p = righ.cols;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            result.mat[i][j] = 0;
            for (int k = 0; k < n; k++) {
                result.mat[i][j] += left.mat[i][k] * righ.mat[k][j];
            }
        }
    }
}

void sub(const matrix& left, const matrix& righ, const matrix& result){
    for (int ri = 0; ri < left.rows; ri++) {
        for (int ci = 0; ci < left.cols; ci++) {
            result.mat[ri][ci] = left.mat[ri][ci] - righ.mat[ri][ci];
        }
    }
}
void add(const matrix& left, const matrix& righ, const matrix& result){
    for (int ri = 0; ri < left.rows; ri++) {
        for (int ci = 0; ci < left.cols; ci++) {
            result.mat[ri][ci] = left.mat[ri][ci] + righ.mat[ri][ci];
        }
    }
}
void sum_col(const matrix& left, const matrix& result){
    // a*b  a, 1
    for (int ri = 0; ri < left.rows; ri++) {
        float sum_ = 0;
        for (int ci = 0; ci < left.cols; ci++) {
            sum_ += left.mat[ri][ci];
        }
        result.mat[ri][0] = sum_;
    }
}
void sum_row(const matrix& left, const matrix& result){
    // a*b  1, a
    for (int ci = 0; ci < left.cols; ci++) {
        float sum_ = 0;
        for (int ri = 0; ri < left.rows; ri++) {
            sum_ += left.mat[ri][ci];
        }
        result.mat[0][ci] = sum_;
    }
}

void copy_matrix(const matrix& from_mat, matrix& to_mat){
    for(int i=0;i<from_mat.rows;i++){
        for(int j=0;j<from_mat.cols;j++){
            to_mat.mat[i][j] = from_mat.mat[i][j];
        }
    }
}


float sigmoid_(float z){
    return 1/(1+exp(-z));
}

void sigmoid(matrix& out){
    for(int ri=0; ri < out.rows; ri++){
        for(int ci=0; ci < out.cols; ci++){
            out.mat[ri][ci] = sigmoid_(out.mat[ri][ci]);
        }
    }
}

void pow2(matrix& out){
    for(int ri=0; ri < out.rows; ri++){
        for(int ci=0; ci < out.cols; ci++){
            out.mat[ri][ci] = out.mat[ri][ci] * out.mat[ri][ci];
        }
    }
}

void mul_const(matrix& out, const float& c){
    for(int ri=0; ri < out.rows; ri++){
        for(int ci=0; ci < out.cols; ci++){
            out.mat[ri][ci] = out.mat[ri][ci] * c;
        }
    }
}

void get_col(const matrix& inp, const int ci, const matrix& out){
    for(int ri=0; ri < out.rows; ri++){
        out.mat[ri][ci] = out.mat[ri][ci];
    }
}

void init_matirx(matrix& ret, int rows, int cols){
    float **mat = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        mat[i] = (float *)malloc(cols * sizeof(int));
    }
    ret.mat = mat;
    ret.rows = rows;
    ret.cols = cols;
}

void forward_to_hidden(const matrix& weights, const matrix& inp, matrix& out){
    dot(weights, inp, out);
    sigmoid(out);
}

void forward_to_output(const matrix& weights, const matrix& inp, matrix& out){
    dot(weights, inp, out);
}


void transpose(const matrix& inp, matrix& out) {
    for (int i = 0; i < inp.cols; i++) {
        for (int j = 0; j < inp.rows; j++) {
            out.mat[i][j] = inp.mat[j][i];
        }
    }
}


void ex(const matrix& inp, matrix& out){
    for(int ri=0; ri < out.rows; ri++){
        for(int ci=0; ci < out.cols; ci++){
            if(ri == inp.rows){
                out.mat[ri][ci] = 1;
            }else{
                out.mat[ri][ci] = inp.mat[ri][ci];
            }
        }
    }
}


// 随机float
float rand_float(float start, float end){
    float s = 1.*rand() / RAND_MAX;
    float ret = (end - start) * s + start;
    return ret;
};

void init_weights(matrix& w){
    for(int ri=0; ri < w.rows; ri++){
        for(int ci=0; ci < w.cols; ci++){
            w.mat[ri][ci] = rand_float(0, 1);
        }}
}




void nn(const matrix& ajsdhjasdl, const matrix& asdhuas){

    int hidden = 2;
    int x_dims = ajsdhjasdl.cols;
    int nums = ajsdhjasdl.rows;
    int y_dims = asdhuas.cols;

    matrix Xt;
    init_matirx(Xt, x_dims, nums);
    transpose(ajsdhjasdl, Xt);
    matrix Yt;
    init_matirx(Yt, y_dims, nums);
    transpose(asdhuas, Yt);

    matrix h1_ex;
    init_matirx(h1_ex, x_dims+1, nums);
    ex(Xt, h1_ex);
    
    matrix w1;
    init_matirx(w1, hidden, x_dims+1);
    init_weights(w1);

    matrix h2;
    init_matirx(h2, hidden, nums);


    forward_to_hidden(w1, h1_ex, h2);
    print_matrix(h2);

    
    matrix h2_ex;
    init_matirx(h2_ex, hidden+1, nums);
    ex(h2, h2_ex);
    matrix w2;
    init_matirx(w2, y_dims, hidden+1);
    init_weights(w2);
    matrix h3;
    init_matirx(h3, y_dims, nums);


    forward_to_output(w2, h2_ex, h3);
    print_matrix(h3);

    matrix error;
    init_matirx(error, y_dims, nums);

    sub(Yt, h3, error);
    print_matrix(error);

    // copy_matrix(error, error2);
    // pow2(error2);
    // print_matrix(error2);

    // matrix dh3;
    // init_matirx(dh3, y_dims, nums);
    // copy_matrix(error, dh3);
    // mul_const(dE, -2);
    // print_matrix(dE);

    
    matrix dEdW2;
    init_matirx(dEdW2, y_dims, hidden+1);
    
    matrix errorT;
    init_matirx(errorT, nums, y_dims);
    transpose(error, errorT);
    print_matrix(errorT);

    // matrix h2_ex_1_col;
    // init_matirx(h2_ex_1_col, hidden+1, 1);
    // get_col(h2_ex, 0, h2_ex_1_col);

    // print_matrix(h2_ex_1_col);
    dot(error, h3, dEdW2);
    // print_matrix(dEdW2);


}


int main(){

    float X_[3][2] = {{1, 1}, {1.01, 1.01}, {-1, -1}};
    float Y_[3][2] = {{1, 0}, {1, 0}, {0, 1}};

    matrix X;
    float* ma_[3];
    for(int i=0;i<3;i++){ma_[i] = X_[i];}
    X = {ma_, 3, 2};

    matrix Y;
    float* mb_[3];
    for(int i=0;i<3;i++){mb_[i] = Y_[i];}
    Y = {mb_, 3, 2};

    // print_matrix(ma);
    // print_matrix(mb);
    // print_matrix(mc);

    // matrix s;
    // init_matirx(s, 2, 2);
    // copy_matrix(Y, s);
    // sigmoid(s);
    // print_matrix(s);

    // matrix s;
    // init_matirx(s, 3, 1);
    // sum_col(X, s);
    // print_matrix(s);
    // matrix s;
    // init_matirx(s, 1, 2);
    // sum_row(X, s);
    // print_matrix(s);
    // matrix s;
    // init_matirx(s, 3, 2);
    // sub(X, X, s);
    // print_matrix(s);
    // matrix s;
    // init_matirx(s, 3, 2);
    // copy_matrix(X, s);
    // pow2(s);
    // print_matrix(s);
    // matrix s;
    // init_matirx(s, 3, 2);
    // copy_matrix(X, s);
    // mul_const(s, 2.);
    // print_matrix(s);

    nn(X, Y);


    return 0;
}




















