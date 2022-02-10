#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define _USE_MATH_DEFINES
#define n 10000000
int main() {
    float sum = 0;
    float *sin_m = (float*)malloc(sizeof(float) * n);
    #pragma acc kernels
    {
    #pragma acc data copyin(sin_m)
    {
    for(int i = 0; i < n; i++){
        sin_m[i] = sinf(i*2*M_PI/(float)n);
    }
}
}
    for(int i = 0; i < n; i++){
        sum += sin_m[i];
    }

    printf("%e\n", sum);

    return 0;
}
