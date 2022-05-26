#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>


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
__global__ void calf_diff(double* U_d_n, double* U_d, double* U_d_diff, int N, int y_start, int y_end) {
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

    MPI_Status status;
    int my_rank, my_size;

    MPI_Init(&argc, &argv);

    double min_error = 0.000001;
    int N = 1024;
    int iter_max = 50000;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //current proc
    MPI_Comm_size(MPI_COMM_WORLD, &my_size); //proc num

    double *tmp = NULL;
    double *U_d = NULL;
    double *U_d_n = NULL;
    double *tmp_d = NULL;
    double *max_d_d = NULL;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cudaSetDevice(my_rank % my_size);

    int isLast = (my_rank / (my_size - 1));
    int isFirst = (my_size - my_rank) / my_size;

    int start = ((N / my_size) * my_rank) * N;
    int end = (N / my_size * (my_rank + 1) + (N % my_size) * isLast) * N; // [start; end)

    int elemsAmount = end - start;
    int rowAmount = elemsAmount / N;

    tmp = (double*)calloc(elemsAmount, sizeof(double));

    double delta = (double)(20.0 - 10.0) / ((double)N - 1.0);
    if (isFirst)
    {
        // boundaries
        //up
        tmp[0] = 10.0;
        tmp[N - 1] = 20.0;

        for (int i = 1; i < N - 1; i++)
            tmp[i] = tmp[i - 1] + delta;

    }
    if (isLast)
    {
        // low
        tmp[(rowAmount - 1) * N] = 20.0;
        tmp[rowAmount * N - 1] = 30.0;
        // [rowsAmount - 1][127] == [(rowsAmount - 1)*N + (N - 1)] == [rowsAmount * N - 1] -- calculation on edge
        for (int i = (rowAmount - 1) * N + 1; i < rowAmount * N - 1; i++)
            tmp[i] = tmp[i - 1] + delta;
    }

    // left and right
    for (int i = 0 + isFirst; i < rowAmount - isLast; i++) {
        tmp[i * N + 0] = 10.0 + delta * ((start / N) + i);
        tmp[i * N + (N - 1)] = 20.0 + delta * ((start / N) + i);
    }

    cudaMalloc((void **)&U_d, (elemsAmount + 2 * N) * sizeof(double));
    cudaMalloc((void **)&U_d_n, (elemsAmount + 2 * N) * sizeof(double));
    double* Z = (double*)calloc(elemsAmount + 2 * N, sizeof(double));
    // just in case, firstly filling device arrays by zeroes
    cudaMemcpy(U_d, Z, (elemsAmount + 2 * N) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(U_d_n, Z, (elemsAmount + 2 * N) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(U_d + N, tmp, elemsAmount * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(U_d_n + N, tmp, elemsAmount * sizeof(double), cudaMemcpyHostToDevice);

    free(tmp);

    dim3 GS = dim3(16, 16);
    dim3 BS = dim3(ceil(N / (double)GS.x), ceil((rowAmount + 2) / (double)GS.y));

    cudaMalloc(&tmp_d, sizeof(double) * elemsAmount);
    cudaMalloc(&max_d_d, sizeof(double));

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d_d, rowAmount * N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    int topProcess = (my_rank + 1) % my_size;
    int bottomProcess = (my_rank + my_size - 1) % my_size;


    // boundaries
    int row_start = isFirst;  //if first, we move one row down
    int row_end = rowAmount + 1 - isLast; //if last, we move one row up

    int iter = 0;
    double err = 42; // main answer in universe
    double local_err = 0;

    while (err > min_error && iter < iter_max)
    {
        iter += 1;

        MPI_Sendrecv(U_d + N, N, MPI_DOUBLE, bottomProcess, my_rank,
                     U_d + elemsAmount + N, N, MPI_DOUBLE, topProcess, topProcess,
                     MPI_COMM_WORLD, &status);

        MPI_Sendrecv(U_d + elemsAmount, N, MPI_DOUBLE, topProcess, my_rank,
                     U_d, N, MPI_DOUBLE, bottomProcess, bottomProcess,
                     MPI_COMM_WORLD, &status);

        calc_heat_equation<<<BS, GS>>>(U_d_n, U_d, N, row_start, row_end);
        if (iter % 100 == 0)
        {
            calf_diff<<<BS, GS>>>(U_d_n, U_d, tmp_d, N, row_start, row_end);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d_d, rowAmount * N);

            cudaMemcpy(&local_err, max_d_d, sizeof(double), cudaMemcpyDeviceToHost);
            MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            if (isFirst)
                printf("iter: %d error: %e\n", iter, err);
        }
        // swap
        double *tmp = U_d;
        U_d = U_d_n;
        U_d_n = tmp;
    }

    cudaFree(U_d);
    cudaFree(U_d_n);
    cudaFree(tmp_d);
    cudaFree(max_d_d);

    MPI_Finalize();

    return 0;
}
