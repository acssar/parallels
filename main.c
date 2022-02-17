#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define n 128

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
    while (it_num < 1e+6) {
        int flag = 0;
        for(int j = 1; j < n - 1; j++) {
            for (int i = 1; i < n - 1; i++) {
                up[j][i] = u[j][i] + tau * a * (u[j-1][i]
                         - 2 * u[j][i] + u[j + 1][i]) / (h * h)
                                 + tau * a *(u[j][i-1] - 2 * u[j][i] + u[j][i+1])/(h*h) ;
                if (fabs(u[j][i] - up[j][i]) > 1e-6)
                    flag = 1;
                // printf("%e\n", u[i]);
            }
        }
        for (int j = 1; j < n - 1; j++) {
            for (int i = 1; i < n - 1; i++) {
                u[j][i] = up[j][i];
            }
        }
        it_num ++;
        if(flag == 0)
            break;
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
