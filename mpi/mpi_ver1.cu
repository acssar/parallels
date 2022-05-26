#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <mpi.h>
#include <unistd.h>
#define N 11

dim3 BS1, GS1, BS2, GS2;
int max_iters = 1000000, 
    iter = 0;    
double tol = 0.000001,
       *d_out;
void * d_temp_storage = NULL;
size_t temp_storage_bytes = 0;
bool first = true;
double *a,
       *da,
       *tmp,
       *newa,
       *dnewa,
       *ddiff,
       err = tol+1;

__global__ void funk(double * a, double * newa){
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

      if((x>0)&&(x<N-1)&&(y>0)&&(y<N-1))
          newa[x*N+y] = 0.25 * (a[(x+1)*N + y] + a[(x-1)*N + y] + a[x*N + y + 1] + a[x*N + y - 1]);
  
}


__global__ void foo(int rank, int d){
  printf("rank %d uses device = %d\n", rank, d);
}
 

void init(int d, int num_threads){
  size_t size_ = (N*ceil(float(N)/float(num_threads))+N*2)*sizeof(double);
  cudaMallocHost( &a, size_ ); 
  usleep(d*40000);
  int h = ceil(float(N)/float(num_threads)),
      start = d * h + 1,
      end = (d+1) * h;
      if ( end > N ) {
        end = N;
      }
      printf("d: %d, start: %d, end: %d, h: %d, d = %d, num_threads = %d\n\n",d, start, end, h, d, num_threads);
  if(h*d <= N){
  cudaMalloc( &da,  size_ );
  
  cudaMallocHost( &newa,  size_ ); 

  cudaMalloc(&ddiff,  size_ );

  cudaMalloc(&dnewa,  size_ );

  double * ddiff_ = (double*)malloc( size_ );
  
  if ( d == 0 ) {
    a[0]=10.0;
    a[N-1]=20;  
    newa[0]=10.0;
    newa[N-1]=20;
  }
  // if ( d == num_threads - 1 ) {
    if((start<= N) && (end>= N)){
    a[(end-start)*N]=20.0;
    a[(end-start)*N + N-1]=30;
    newa[(end-start)*N]=20.0;
    newa[(end-start)*N+N-1]=30;
  }
  
  double  d1=(10.0)/(N-1);
          
  for (int i = 1; i < N; i++){
    if(d==0){
      a[i]=10 + d1*i;
      newa[i]=10 + d1*i;
    }
    if(d == num_threads-1){
      a[(N-start-1)*N + i]=20 + d1*i;
      newa[(N-start-1)*N + i]=20 + d1*i;
    }
    if(i>=start && i<=end){

      a[(i-start)*N]=10 + d1*i;

      a[(i-start)*N+N-1]=20 + d1*i;

      
     

      newa[(i-start)*N]=10 + d1*i;

      newa[(i-start)*N+N-1]=20 + d1*i;

      

    }
  }

  MPI_Rsend(void* message, int count,
    MPI_Datatype datatype, int dest, int tag,
    MPI_Comm comm);
  
int MPI_Recv(void* message, int count,
    MPI_Datatype datatype, int source, int tag,
    MPI_Comm comm, MPI_Status* status);

  usleep((d+1) * 500000 );
  for (int i = start-1; i <= end + 1; i ++){
    printf("\n");
    for (int j = 0; j < N; j ++){
      if (j==0)
        if((i<start)||(i>end))
          printf("-%d ", i);
        else
          printf("d%d ", i);
      printf("%f ", a[(i-start)*N + j]);
      }
    }
    printf("\n");
}
}
// int **gg;
// void test(int *gg[]){
//   for (int i = 0; i < 5; i ++){
//     printf("\n");
//     for(int j = 0; j < 5; j++)
//       printf("%d ", gg[i][j]);

//   }
// }

int main(int argc, char *argv[]) {
  // gg = (int**)malloc(sizeof(int*)*5);
  // for (int i = 0; i < 5; i ++){
  //   gg[i]=(int*)malloc(sizeof(int)*5);
  // }
  // for (int i = 0; i < 5; i ++){

  //   for(int j = 0; j < 5; j++)
  //     gg[i][j] = i*j;
  // }

  // test(gg);
  
  int rank,size, d;
    /* Initialize the MPI library */
    MPI_Init(&argc, &argv);
    
    /* Determine the calling process rank and total number of ranks */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    /* Call MPI routines like MPI_Send, MPI_Recv, ... */
  
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(rank % num_devices);
    cudaError_t err = cudaGetDevice(&d);
    if (err != cudaSuccess) printf("kernel err\n");
    //foo<<<1, 1>>>(rank,d); 
      init(rank, size);
    cudaDeviceSynchronize();
   
    /* Shutdown MPI library */
    MPI_Finalize();

    return 0;
} 
