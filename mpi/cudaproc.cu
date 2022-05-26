#include <cuda.h>
#include <cub/cub.cuh>
#include <math.h>


__global__ void iteration(double *a, double *b, int start, int end, int size_x, int size_y)
{
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int idy = blockIdx.y*blockDim.y + threadIdx.y;

        // Does it belong to this GPU area
        if(idx*size_x + idy >= start && idx*size_x + idy < end)
            // Is it in array borders
            if(idx < size_x - 1 && idy < size_y - 1 && idx >= 1 && idy >= 1)
                a[idx*size_x + idy] = (b[(idx+1)*size_x + idy] + b[(idx-1)*size_x + idy] + b[idx*size_x + (idy + 1)] + b[idx*size_x + (idy - 1)]) / 4;

        __syncthreads();

        if(idx*size_x + idy >= start && idx*size_x + idy < end)
            if(idx < size_x - 1 && idy < size_y - 1 && idx >= 1 && idy >= 1)
                b[idx*size_x + idy] = (a[(idx+1)*size_x + idy] + a[(idx-1)*size_x + idy] + a[idx*size_x + (idy + 1)] + a[idx*size_x + (idy - 1)]) / 4;
        __syncthreads();

    return;
}


__global__ void initMatrix(double *mass1, double *mass2, int size)
{
    mass1[0] = 10;
    mass1[(size-1)*size +size-1] = 30;
    mass1[size-1] = 20;
    mass1[(size-1)*size] = 20;

    mass2[0] = 10;
    mass2[(size-1)*size +size-1] = 30;
    mass2[size-1] = 20;
    mass2[(size-1)*size] = 20;

    for(int i = 1; i < size - 1; ++i)
    {
        mass1[i] = 10.0 + 10.0 * i/(size - 1.0);
        mass1[i*size] = 10.0 + 10.0 * i / (size - 1.0);
        mass1[i*size + size-1] = 20.0 + 10.0 * i / (size - 1.0);
        mass1[(size-1)*size + i] = 20.0 + 10.0 * i / (size - 1.0);
        mass2[i] = 10.0 + 10.0 * i/(size - 1.0);
        mass2[i*size] = 10.0 + 10.0 * i / (size - 1.0);
        mass2[i*size + size-1] = 20.0 + 10.0 * i / (size - 1.0);
        mass2[(size-1)*size + i] = 20.0 + 10.0 * i / (size - 1.0);
    }
}


int get_device_var(int GRID, double *dev_mass, double *dev_mass_plus, double *buff)
{
    cudaMalloc((void**)&dev_mass, sizeof(double) * GRID * GRID);
    cudaMalloc((void**)&dev_mass_plus, sizeof(double) * GRID * GRID);

    initMatrix<<< 1, 1 >>> (dev_mass_plus, dev_mass, GRID);
    cudaMalloc((void**)&buff, sizeof(double) * GRID);

    return 0;
}

int get_cpu_var(int GRID, double *mass, double *dev_mass)
{
    mass = (double*) calloc(GRID*GRID, sizeof(double));
    cudaMemcpy(mass, dev_mass, sizeof(double), cudaMemcpyDeviceToHost);
    return 0;
}

int iter(double *a, double *b, int start, int end, int size_x, int size_y)
{

    const dim3 BS = dim3(size_x/8, size_y/8);
    const dim3 GS = dim3(ceil(size_x/(float)BS.x), ceil(size_y/(float)BS.y));

    iteration <<< BS, GS >>> (a, b, start, end, size_x, size_y);

    return 0;
}
       
