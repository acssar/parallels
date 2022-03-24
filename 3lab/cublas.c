#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/math_libs/11.5/targets/x86_64-linux/include/cublas_v2.h"

#define N 128

int main(void)
{
    double* copy_pointer;
    int it_num = 0;
    double* err = (double*)calloc(1, sizeof(double));
    err[0] = 10;
    double step = 10.0 / (N - 1);
    cublasHandle_t handle;
    cublasCreate(&handle);

    double* u = (double*)calloc(N * N, sizeof(double));
    double* up = (double*)calloc(N * N, sizeof(double));

#pragma acc enter data create(u[0:N*N], up[0:N*N]) copyin(N, step)
#pragma acc kernels
    {
#pragma acc loop independent
        for (int i = 0; i < N; i++) {
            u[i * N] = 10 + step * i;
            u[i] = 10 + step * i;
            u[(N - 1) * N + i] = 20 + step * i;
            u[i * N + N - 1] = 20 + step * i;

            up[i * N] = u[i * N];
            up[i] = u[i];
            up[(N - 1) * N + i] = u[(N - 1) * N + i];
            up[i * N + N - 1] = u[i * N + N - 1];
        }
    }


    {
        while (err[0] > 1e-6 && it_num < 10000) {

            it_num++;

            if (it_num % 100 == 0) {
#pragma acc data present(u[0:N*N], up[0:N*N])
#pragma acc kernels async(1)
                {
#pragma acc loop independent collapse(2)
                    for (int i = 1; i < N - 1; i++)
                        for (int j = 1; j < N - 1; j++)
                            up[i * N + j] = 0.25 * (u[(i + 1) * N + j] + u[(i - 1) * N + j] + u[i * N + j - 1] + u[i * N + j + 1]);
                }
                int idx = 0;
#pragma acc wait
#pragma acc host_data use_device(u, up)
                {
                    const double alpha = -1;
                    cublasDaxpy(handle, N * N, &alpha, up, 1, u, 1);
                    cublasIdamax(handle, N*N, u, 1, &idx);
                }

#pragma acc update self(u[idx - 1:1])
                err[0] = fabs(u[idx - 1]);
#pragma acc host_data use_device(u, up)
                cublasDcopy(handle, N*N, up, 1, u, 1);
            }
            else {

#pragma acc data present(u[0:N*N], up[0:N*N])
#pragma acc kernels async(1)
                {
#pragma acc loop independent collapse(2)
                    for (int i = 1; i < N - 1; i++)
                        for (int j = 1; j < N - 1; j++)
                            up[i * N + j] = 0.25 * (u[(i + 1) * N + j] + u[(i - 1) * N + j] + u[i * N + j - 1] + u[i * N + j + 1]);
                }
            }


            copy_pointer = u;
            u = up;
            up = copy_pointer;

            if (it_num % 100 == 0) {
#pragma acc wait(1)
                printf("%d %lf\n", it_num, err[0]);
            }
        }
    }

    cublasDestroy(handle);
    printf("%d %lf\n", it_num, err[0]);



    return 0;
}
