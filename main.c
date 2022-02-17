#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define n 128

int main() {

    double u[n], up[n];
    u[0] = 0.0;  //для итерации n-1 массив
    for (int i = 1; i < n-1; i++){
        u[i] = 0.0;
    }
    u[n-1] = 1.0;
    up[0] = 0.0;
    up[n-1] = 1.0;
    double x_max = 1.0;
    double h = x_max/(double)n;
    double a = 1.0; //коэфф теплопроводности
    double tau = a / (n*n*n); //tau - временной шаг, должен быть меньше h = 1/n
    //итерации, пока не будет разница up - u < 10^-6

    int it_num = 0;
    while (1) {
        int flag = 0;
        for (int i = 1; i < n - 1; i++) {
            up[i] = u[i] + tau * a * (u[i - 1] - 2 * u[i] + u[i + 1]) / (h*h);
            if(fabs(u[i] - up[i]) > 1e-6)
                flag = 1;
           // printf("%e\n", u[i]);
        }
        for (int i = 1; i < n - 1; i++) {
            u[i] = up[i];
        }
        it_num ++;
        if(flag == 0)
            break;
    }
    for (int i = 0; i < n; i++)
        printf("%e\n", u[i]);
    printf("number of iterations: %d", it_num);

    return 0;
}
