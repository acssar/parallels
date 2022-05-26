#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>
#include <nccl.h>

__global__ void calc_heat_equation(double* U_d_n, double *U_d, int N, int y_start, int y_end) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < y_end && j > y_start) {
        if (i > 0 && i < N - 1) {
            double left = U_d[j * N + i - 1];
            double right = U_d[j * N + i + 1];
            double up = U_d[(j - 1) * N + i];
            double down = U_d[(j + 1) * N + i];

            U_d_n[j * N + i] = 0.25 * (left + right + up + down);

        }
    }
}

__global__ void calc_diff(double* U_d_n, double* U_d, double* U_d_diff, int N, int y_start, int y_end) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N - 1) {
        if(j > y_start && j < y_end){
            U_d_diff[j * N + i] = U_d_n[j * N + i] - U_d[j * N + i];
        }
    }
}

int main(int argc, char * argv[])
{


    int my_rank, my_size;

    MPI_Init(&argc, &argv);

    double min_err = 0.000001;
    int N = 1024;
    int iter_max = 50000;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &my_size);
    ncclUniqueId n_id;

    double *tmp = NULL;
    double *U_d = NULL;
    double *U_d_n = NULL;
    double *tmp_d = NULL;
    double *max_d = NULL;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;


    int isLast = -1; //define which proc is last and first
    int isFirst = -1;
    if(my_size > 1){
        isLast = (my_rank / (my_size - 1));
        isFirst = (my_size - my_rank) / my_size;
    }
    if(my_size == 1)
    {
        isLast = 1;
        isFirst = 1;
    }
    if(isFirst)
        ncclGetUniqueId(&n_id);

    MPI_Bcast(&n_id, sizeof(n_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaSetDevice(my_rank % my_size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int start = ((N / my_size) * my_rank) * N;
    int end = (N / my_size * (my_rank + 1) + (N % my_size) * isLast) * N; // [start; end)

    int elemsAmount = end - start;
    int rowsAmount = elemsAmount / N;

    // interpolation
    tmp = (double*)calloc(elemsAmount, sizeof(double));

    double delta = (double)(20.0 - 10.0) / ((double)N - 1.0);
    if (isFirst)
    {
        // up
        tmp[0] = 10.0;
        tmp[N - 1] = 20.0;

        for (int i = 1; i < N - 1; i++)
            tmp[i] = tmp[i - 1] + delta;

    }
    if (isLast)
    {
        //  low
        tmp[(rowsAmount - 1) * N] = 20.0;
        tmp[rowsAmount * N - 1] = 30.0;
        // [rowsAmount - 1][127] == [(rowsAmount - 1)*N + (N - 1)] == [rowsAmount * N - 1]  reminder
        for (int i = (rowsAmount - 1) * N + 1; i < rowsAmount * N - 1; i++)
            tmp[i] = tmp[i - 1] + delta;
    }

    // left, right
    for (int i = 0 + isFirst; i < rowsAmount - isLast; i++) {
        tmp[i * N + 0] = 10.0 + delta * ((start / N) + i);
        tmp[i * N + (N - 1)] = 20.0 + delta * ((start / N) + i);
    }
    // copying to GPU
    cudaMalloc((void **)&U_d, (elemsAmount + 2 * N) * sizeof(double));
    cudaMalloc((void **)&U_d_n, (elemsAmount + 2 * N) * sizeof(double));
    double* Z = (double*)calloc(elemsAmount + 2 * N, sizeof(double));
    // just in case fill with zeros first
    cudaMemcpyAsync(U_d, Z, (elemsAmount + 2 * N) * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(U_d_n, Z, (elemsAmount + 2 * N) * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(U_d + N, tmp, elemsAmount * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(U_d_n + N, tmp, elemsAmount * sizeof(double), cudaMemcpyHostToDevice, stream);

    free(tmp);

    dim3 GS = dim3(16, 16);
    dim3 BS = dim3(ceil(N / (double)GS.x), ceil((rowsAmount + 2) / (double)GS.y));

    cudaMalloc(&tmp_d, sizeof(double) * elemsAmount);
    cudaMalloc(&max_d, sizeof(double));

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d, rowsAmount * N, stream);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    int topProcess = (my_rank + 1) % my_size;
    int bottomProcess = (my_rank + my_size - 1) % my_size;


    // starting rows for process moved for 1st and last
    int y_start = isFirst;
    int y_end = rowsAmount + 1 - isLast;

    int it = 0;
    double err = 42; //answer to question of Universe
    double * err_d;
    cudaMalloc(&err_d, sizeof(double));

    ncclComm_t n_comm;
    ncclCommInitRank(&n_comm, my_size, n_id, my_rank);
//    printf("Rank %d\n", my_rank);
    while (err > min_err && it < iter_max)
    {
        it ++;
        if(my_size > 1)
        {   //almost as mpi
            ncclGroupStart();
            ncclSend(U_d + N, N, ncclDouble, bottomProcess, n_comm, stream);
            ncclSend(U_d + elemsAmount, N, ncclDouble, topProcess, n_comm, stream);

            ncclRecv(U_d + elemsAmount + N, N, ncclDouble, topProcess, n_comm, stream);
            ncclRecv(U_d, N, ncclDouble, bottomProcess, n_comm, stream);
            ncclGroupEnd();
        }

        calc_heat_equation<<<BS, GS,0, stream>>>(U_d_n, U_d, N, y_start, y_end);
        if (it % 100 == 0)
        {
            calc_diff<<<BS, GS, 0, stream>>>(U_d_n, U_d, tmp_d, N, y_start, y_end);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d, rowsAmount * N, stream);
            ncclAllReduce(max_d, err_d, 1, ncclDouble, ncclMax, n_comm, stream);
            cudaMemcpyAsync(&err, err_d, sizeof(double), cudaMemcpyDeviceToHost, stream);
            if (isFirst)
                printf("iter: %d error: %e\n", it, err);
        }
        //swap
        double *tmp = U_d;
        U_d = U_d_n;
        U_d_n = tmp;
    }

    cudaFree(U_d);
    cudaFree(U_d_n);
    cudaFree(tmp_d);
    cudaFree(max_d);

    cudaStreamDestroy(stream);
    ncclCommDestroy(n_comm);
    MPI_Finalize();

    return 0;
}
