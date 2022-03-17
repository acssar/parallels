#include <stdio.h>
#include <math.h>
#include <malloc.h>
#define  N 128
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/math_libs/11.5/targets/x86_64-linux/include/cublas_v2.h"

int main() {
    double **u = (double **) calloc(N, sizeof(double *)); // first

    for (int i = 0; i < N; i++) {
        u[i] = (double *) calloc(N, sizeof(double));
    }

    double **up = (double **) calloc(N, sizeof(double *));  // second

    for (int j = 0; j < N; j++) {
        up[j] = (double *) calloc(N, sizeof(double));
    }

    u[0][0] = 10.0;
    u[N - 1][0] = 20.0;
    u[0][N - 1] = 20.0;
    u[N - 1][N - 1] = 30.0;

    up[0][0] = 10.0;
    up[N - 1][0] = 20.0;
    up[0][N - 1] = 20.0;
    up[N - 1][N - 1] = 30.0;

    double step = 10.0 / (N - 1);
    int it_num = 0;
    double err = 1;

    cublasHandle_t handle;
    cublasCreate(&handle);
    double temp[N*N];

    //interpolation
    for (int k = 1; k < N - 1; k++) {
        u[k][0] = 10 + step * k;
        u[0][k] = 10 + step * k;
        u[k][N - 1] = 20 + step * k;
        u[N - 1][k] = 20 + step * k;
        up[k][0] = 10 + step * k;
        up[0][k] = 10 + step * k;
        up[k][N - 1] = 20 + step * k;
        up[N - 1][k] = 20 + step * k;

    }

#pragma acc data copy(u[0:N][0:N]) create (up[0:N][0:N], err, temp)
    {
        while (err > 0.000001 && it_num < 1000000) {

            it_num++;
           // if(it_num % 100 == 0 ){
	   //
#pragma acc kernels async(1)
	    {    
	    	err = 0;
            if (it_num % 100 == 0 || it_num == 1) {
#pragma acc loop independent collapse(2)

            
                
                    for (int i = 1; i < N - 1; i++) {

                        for (int j = 1; j < N - 1; j++) {

                            up[i][j] = 0.25 * (u[i + 1][j] + u[i - 1][j] + u[i][j - 1] + u[i][j + 1]);
                            //err = fmax(err, up[i][j] - u[i][j]);
                        }
                    }
                }
#pragma acc wait(1)
int idx;	
#pragma acc host_data use_device(up, u, temp)
{
	double alpha = -1.;
	for (int i = 0; i < N; i++){
		cublasDcopy_v2(handle, N, up[i],1, &temp[i*N], 1);
		cublasDaxpy_v2(handle, N, &alpha, u[i],1, &temp[i*N], 1);
	}

	cublasIdamax_v2(handle, N*N, temp, 1, &idx);
}
#pragma acc update self(temp)
err = temp[idx-1];
	    else{
#pragma acc loop independent collapse(2)
		    {
                    for (int i = 1; i < N - 1; i++) {
                        for (int j = 1; j < N - 1; j++) {

                            up[i][j] = 0.25 * (u[i + 1][j] + u[i - 1][j] + u[i][j - 1] + u[i][j + 1]);
                        }
                    }		    
	    }
	    }
#pragma acc parallel loop independent collapse(2) async(1)                
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {

                    u[i][j] = up[i][j];
                }
            }

            if(it_num % 100 == 0 || it_num == 1 )
#pragma acc wait(1)
#pragma acc update self(err)
                printf("%d %e\n", it_num, err);
            


        }
    }


    return 0;
}
