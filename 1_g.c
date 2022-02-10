#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define _USE_MATH_DEFINES
#define n 10000000
int main() {
    double sum = 0;
    double *sin_m = (double*)malloc(sizeof(double) * n);
    #pragma acc kernels
    {
    #pragma acc data copyin(sin_m)
    {
    for(int i = 0; i < n; i++){
        sin_m[i] = sin(i*2*M_PI/(double)n);
    }
}
}
    for(int i = 0; i < n; i++){
        sum += sin_m[i];
    }

    printf("%e\n", sum);

    return 0;
}
