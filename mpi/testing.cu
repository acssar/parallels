#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <mpi.h>
#define N 128



__global__ void iteration(double *a, double *b, int start, int end, int size_x, int size_y)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;

    // Does it belong to this GPU area
    if(idx*size_x + idy >= start && idx*size_x + idy < end)
        // Is it in array borders
        if(idx < size_x - 1 && idy < size_y - 1 && idx >= 1 && idy >= 1)
            a[idx*size_x + idy] = (b[(idx+1)*size_x + idy] + b[(idx-1)*size_x + idy] + b[idx*size_x + (idy + 1)] + b[idx*size_x + (idy - 1)]) / 4;

}

