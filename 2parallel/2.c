#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define n 512

int main() {
    FILE *f;
    f = fopen("f.txt", "wt");
    double u[n][n], up[n][n]; //для итерации n-1 массив
    for (int j = 1; j < n-1; j++){
        for (int i = 1; i < n-1; i++) {
            u[j][i] = 0.0;
        }
    }
    u[0][0] = 10.0;
    u[0][n-1] = 20.0;
    u[n-1][0] = 30.0;
    u[n-1][n-1] = 20.0;
    up[0][0] = u[0][0];
    up[n-1][0] = u[n-1][0];
    up[0][n-1] = u[0][n-1];
    up[n-1][n-1] = u[n-1][n-1];

    double x_max = 1.0;
    double h = x_max/(double)n;
    double a = 1.0; //коэфф теплопроводности
    double tau = 1e-5; //tau - временной шаг, должен быть меньше h = 1/n
    //итерации, пока не будет разница up - u < 10^-6
    double x,y,z,c;
    x = (u[0][n-1] - u[0][0])/(n-2); // шаг первой строки
    y = (u[n-1][0] - u[0][0])/(n-2); //шаг первого столбца
    z = (u[n-1][n-1] - u[n-1][0])/(n-2); //шаг последней строки
    c = (u[n-1][n-1] - u[0][n-1])/(n-2); // шаг последнего столбца

    for (int i = 1; i < n -1; i++){
        u[0][i] = u[0][0] + i * x;
        u[i][0] = u[0][0] + i * y;
        u[n-1][i] = u[n-1][0] + i * z;
        u[i][n-1] = u[0][n-1] + i * c;
    }









    int it_num = 0;
    double err = 1;
#pragma acc data copy(u) create(up)
    while (err > 1e-6 && it_num < 100000) {
        err = 0;
#pragma acc data present(u, up)
#pragma acc parallel reduction(max:err)
        {
#pragma acc loop independent
            for (int i = 1; i < n - 1; i++) {
#pragma acc loop independent
                for (int j = 1; j < n - 1; j++) {
                    up[i][j] = 0.25 * (u[i][j - 1] + u[i][j + 1] + u[i + 1][j] + u[i - 1][j]);
                    err = fmax(err, up[i][j] - u[i][j]);
                }
            }
        }
#pragma acc parallel
        {
#pragma acc loop independent
            for (int i = 1; i < n - 1; i++)
#pragma acc loop independent
                    for (int j = 1; j < n - 1; j++)
                        u[i][j] = up[i][j];
        }
	it_num++;
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            fprintf(f,"%10.3e", u[j][i]);
        }
        fprintf(f, "\n");
    }
    fprintf(f,"number of iterations: %d\n", it_num);
    fclose(f);

    return 0;
}

