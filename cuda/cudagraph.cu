#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cub/cub.cuh>
#define N 1024

__global__ void calc_heat_equation(double* U_d, double* U_d_n, int n)
{

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n - 1 && j > 0)
    {
        if (i < n - 1 && i > 0)
        {
            double left = U_d[i * n + j - 1];
            double right = U_d[i * n + j + 1];
            double up = U_d[(i - 1) * n + j];
            double down = U_d[(i + 1) * n + j];

            U_d_n[i * n + j] = 0.25 * (left + right + up + down);
        }
    }
}

__global__ void calc_diff(double* U_d, double* U_d_n, double* U_d_diff, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= 0 && i < n)
    {
        if (j >= 0 && j < n)
        {
        U_d_diff[i * n + j] = U_d_n[i * n + j] - U_d[i * n + j];
        }
    }
}



int main(void)
{

    double* U = (double*)calloc(N*N, sizeof(double));
    double* U_n =(double*)calloc(N*N, sizeof(double));

    double* U_d;
    double* U_d_n;
    double* U_d_diff;

    cudaMalloc(&U_d, sizeof(double)*N*N);
    cudaMalloc(&U_d_n, sizeof(double)*N*N);
    cudaMalloc(&U_d_diff, sizeof(double)*N*N);
    double delta = 10.0 / (N - 1);

    for (int i = 0; i < N; i++)
    {
        U[i*N] = 10 + delta * i;
        U[i] = 10 + delta * i;
        U[(N-1)*N + i] = 20 + delta * i;
        U[i*N + N - 1] = 20 + delta * i;

        U_n[i*N] = U[i*N];
        U_n[i] = U[i];
        U_n[(N-1)*N + i] = U[(N-1)*N + i];
        U_n[i*N + N - 1] = U[i*N + N - 1];
    }


    cudaMemcpy(U_d, U, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(U_d_n, U_n, N*N*sizeof(double), cudaMemcpyHostToDevice);

    int bss = ceil(N/4);
    dim3 BLOCK_SIZE = dim3(32, 32); //размер блока 32 по гориз, 32 по вертик.
    dim3 GRID_SIZE = dim3(ceil(N/32.),ceil(N/32.)); //количество блоков по гориз. верт.

    int it = 0;
    int max_it = 50000;
    double* err = (double*)calloc(1,sizeof(double));
    *err = 1;
    double* d_err;
    cudaMalloc(&d_err, sizeof(double));

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, U_d_diff, d_err, N*N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
   
    int step = 100;
    int max_it_with_graphs = max_it / step;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
	    
    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    while(it < max_it_with_graphs && *err > 1e-6)
    {
        it++;
	if(!graphCreated)
	{
		cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
		for(int i = 0; i < 100; i ++)
		{
			calc_heat_equation<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(U_d, U_d_n, N); //запуск ядер
			double* swap_ptr = U_d;
			U_d = U_d_n;
			U_d_n = swap_ptr;
		}
		cudaStreamEndCapture(stream,&graph);
		cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

		graphCreated=true;
	}
	cudaGraphLaunch(instance, stream);
	cudaStreamSynchronize(stream);
	
	printf("iter = %d error = %e\n", it*step, *err);
	*err = 0;
	double* swap_ptr = U_d;
	U_d = U_d_n;
	U_d_n = swap_ptr;

        calc_diff<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(U_d, U_d_n, U_d_diff, N); //высчитывание массивов разниц
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, U_d_diff, d_err, N*N, stream); //нахождение максимальной разности
        cudaMemcpyAsync(err, d_err, sizeof(double), cudaMemcpyDeviceToHost, stream);
        
        swap_ptr = U_d;
        U_d = U_d_n;
        U_d_n = swap_ptr;
    }
/*
    cudaMemcpy(U_n, U_d_n, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N*N; i ++){
        printf("%lf\n", U_n[i]);
    }
    printf("\\\\\\\\\\");
*/
    free(U);
    free(U_n);
    cudaFree(U_d);
    cudaFree(U_d_n);
    cudaFree(U_d_diff);
    printf("fin = %d %13.10lf\n", it, *err);
    return 0;
}
