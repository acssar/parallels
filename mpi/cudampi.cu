#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <mpi.h>
#define N 128
__global__ void calc_heat_equation(double* U_d, double* U_d_n, int h, int start)
{

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j > 0 && j < N - 1){
        if (i < h + 1 && i > start)
        {
            double left = U_d[i * N + j - 1];
            double right = U_d[i * N + j + 1];
            double up = U_d[(i - 1) * N + j];
            double down = U_d[(i + 1) * N + j];

            U_d_n[i * N + j] = 0.25 * (left + right + up + down);
        }
    }
    
}

__global__ void calc_diff(double* U_d, double* U_d_n, double* U_d_diff, int h)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N * h)
    {
        U_d_diff[index] = U_d_n[index] - U_d[index];   
    }
}



int main(int argc, char* argv[])
{
    int myrank, size, cu_d; // myrank - local rank, size - num of processes
    int num_device;
    MPI_Init(&argc, &argv);

    
    /* Determine unique id of the calling process of all processes participating
       in this MPI program. This id is usually called MPI rank. */
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); //от 1 до текущего кол-ва запущенных процессов

   /* Retrieves the number of processes involved in a communicator, or the total number of processes available.*/
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cudaGetDeviceCount(&num_device);
    cudaSetDevice(myrank % num_device);
	
    int rk = myrank, num_threads = size; 
    int h, start, end, max_rk;
    
    start = 1;
    h = ceil(float(N)/float(num_threads));
    end = h;
    if(end*(d+1) > N)
	    end = (N % h)
	
    max_rk = ceil(float(N)/h) - 1; 
    size_t size_ = (N * ceil(float(N)/float(num_threads)) + N * 2) * sizeof(double);
    double* U = (double*)calloc(N*N, sizeof(double));
    double* U_n =(double*)calloc(N*N, sizeof(double));
    
    double* U_d;
    double* U_d_n;
    double* U_d_diff;

    cudaMalloc(&U_d, sizeof(double)*N*N);
    cudaMalloc(&U_d_n, sizeof(double)*N*N);
    cudaMalloc(&U_d_diff, sizeof(double)*N*N);

    //intepolation of boundaries with different processes

    if (rk == 0) {
        U[start * N] = 10;
	U[(start+1) * N - 1] = 20;
	U_n[start * N] = 10;
	U_n[(start+1) * N - 1] = 20;
    }
	
    if (rk == max_rk){
	U[(end) * N] = 20;
    	U[end * N + N - 1] = 30;
	U_n[end * N] = 20;
	U_n[end * N + N - 1] = 30;
    }
    

    double delta = 10.0 / (N - 1);

    for (int i = 1; i < N; i++)
    {
	if(rk == 0){		
        U[start * N + i] = 10 + delta * i;
	U_n[start * N + i] = 10 + delta * i;
	}
	if(rk == max_rk){
	U[end * N + i] = 20 + delta * i;
 	U_n[end * N + i] = 20 + delta * i;
	}
	
	if(i >= h * rk && i < h * rk + end){
	U[(start + i - h * rk) * N] = 10 + delta * i;
	U[(start + i - h * rk) * N + N - 1] = 20 + delta * i;

	U_n[(start + i - h * rk) * N] = 10 + delta * i;
	U_n[(start + i - h * rk) * N + N - 1] = 20 + delta * i;	
    }
    }

    //copy to GPU
    cudaMemcpy(U_d, U, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(U_d_n, U_n, N*N*sizeof(double), cudaMemcpyHostToDevice);

 
  
    dim3 DBS = 128; // for diff calc
    dim3 DGS = ceil((float)(size_)/(float)(DBS.x));		    
    dim3 BLOCK_SIZE = dim3(16, 16); //размер блока 16 по гориз, 16 по вертик.
    dim3 GRID_SIZE = dim3(ceil(N/16.),ceil((float)(h+2)/16.)); //количество блоков по гориз. верт.

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
   
   // int step = 100;  // from graph
   // int max_it_with_graphs = max_it / step;
    //boundary ranks
    int topProc = myrank !=   /// dodelatttt
    int botProc = myrank != 0 /// dodelatttt

    //cudaStream_t stream;
    //cudaStreamCreate(&stream);

    //limitation of boundaries for processes
    int y_start = myrank == 0 & 1 : 0;
    int y_end = myrank == size - 1 ? (end - start): end - start + 1;    
    

    while(it < max_it && *err > 1e-6)
    {
        it++;
	MPI_Sendrecv(U_d + N, N, MPI_DOUBLE,botProc, myrank, U_d + end - start  	
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
    MPI_Finalize();
    free(U);
    free(U_n);
    cudaFree(U_d);
    cudaFree(U_d_n);
    cudaFree(U_d_diff);
    printf("fin = %d %13.10lf\n", it, *err);
    return 0;
}
