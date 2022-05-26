#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <mpi.h>
#include <unistd.h>

#define N 11
using namespace std;
dim3 BS1, GS1, BS2, GS2;


typedef struct {
  int h, start, end, max_d;
} info;


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
       *ddiff_, 
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
  int h, start, end, max_d;

  h = ceil(float(N)/float(num_threads));

  start = 1;
  end = h;
  if(end*(d+1) > N){
    end = (N % (h));
  }

  max_d = ceil((float)N/h)-1;
 

  size_t size_ = (N*ceil(float(N)/float(num_threads))+N*2)*sizeof(double);
  printf("d: %d, start: %d, end: %d, h: %d, d = %d, num_threads = %d, max_d = %d\n\n",d, start, end, h, d, num_threads, max_d);


  // usleep(d*40000);

  
  if(max_d >= d){

  cudaMallocHost( &a, size_ ); 

  cudaMalloc( &da,  size_ );
  
  cudaMallocHost( &newa,  size_ ); 

  cudaMalloc(&ddiff,  size_ );

  cudaMalloc(&dnewa,  size_ );

  ddiff_ = (double*)malloc( size_ );
  
  if ( d == 0 ) {
    a[start*N]=10.0;
    a[(start+1)*N-1]=20;  
    newa[start*N]=10.0;
    newa[(start+1)*N-1]=20;
  }

  if(d == max_d){
    a[(end)*N]=20.0;
    a[(end)*N + N-1]=30;
    newa[(end)*N]=20.0;
    newa[(end)*N + N-1]=30;
  }
  
  double  d1=(10.0)/(N-1);
        

  for (int i = 1; i < N; i++){
    if(d==0){
      a[start*N + i]=10 + d1*i;
      newa[start*N + i]=10 + d1*i;
    }
    if(d == max_d){
      a[(end)*N + i]=20 + d1*i;
      newa[(end)*N + i]=20 + d1*i;
    }
    
    if(i>=h*d && i< h*d + end){

      a[(start+i - h*d)*N]=10 + d1*i;

      a[(start+i - h*d)*N+N-1]=20 + d1*i;

      
     

      newa[(start+i - h*d)*N]=10 + d1*i;

      newa[(start+i - h*d)*N+N-1]=20 + d1*i;

    }
  }

    MPI_Status* status;
    
    if (max_d != 0)
    {
    if(d == 0){
       MPI_Sendrecv(&a[end*N], N, MPI_DOUBLE, d+1, d,
                   &a[(end+1)*N], N, MPI_DOUBLE, d+1, d+1,
                       MPI_COMM_WORLD, status);
     }
   
   if((d>0)&&(d<max_d)){
       MPI_Sendrecv(&a[(end)*N], N, MPI_DOUBLE, d+1, d,
                     &a[N*(start - 1)], N, MPI_DOUBLE, d-1, d-1,
                       MPI_COMM_WORLD, status);
   
       MPI_Sendrecv(&a[start*N], N, MPI_DOUBLE, d-1 , d,
                     &a[(end+1)*N], N, MPI_DOUBLE, d+1, d+1,
                       MPI_COMM_WORLD, status);
     }
   
     if(d == max_d){
       MPI_Sendrecv(&a[start*N], N, MPI_DOUBLE, d-1 , d,
         &a[(start-1)*N], N, MPI_DOUBLE, d-1, d-1,
           MPI_COMM_WORLD, status);
     }
    }

  //   MPI_Comm comm;
  // MPI_Barrier(comm);
  usleep((d+1) * 500000 );
  // printf("tuta");



  /////////////////////////////////////////////////////////////////////////////////////// PRINT
  for (int i = start-1; i <= end + 1; i ++){
    printf("\n");
    for (int j = 0; j < N; j ++){
      if (j==0)
        if((i<start)||(i>end))
          printf("-%d ", h*d + i);
        else
          printf("+%d ", h*d + i);
      printf("%f ", a[(i)*N + j]);
      }
    }
    printf("\n"); 
  ////////////////////////////////////////////////////////////////////////////////////// END PRINT

  // cudaStream_t stream1, stream2, stream3, stream4;
  // cudaStreamCreate ( &stream1);
  // cudaStreamCreate ( &stream2);
  
  
  // cudaMemcpyAsync( da, a, size_, cudaMemcpyHostToDevice, stream1);
  // cudaMemcpyAsync( dnewa, newa, size_, cudaMemcpyHostToDevice, stream2);
  
  // cudaStreamSynchronize(stream1);
  // cudaStreamSynchronize(stream2);
  // cudaStreamCreate ( &stream4);
  





 }

}

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
  
  int rank,size, cu_d;
    /* Initialize the MPI library */
    MPI_Init(&argc, &argv);
    
    /* Determine the calling process rank and total number of ranks */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    /* Call MPI routines like MPI_Send, MPI_Recv, ... */
  
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(rank % num_devices);
    cudaError_t err = cudaGetDevice(&cu_d);
    if (err != cudaSuccess) printf("kernel err\n");
    //foo<<<1, 1>>>(rank,d); 
   

    // init(rank, size);




    int d = rank, num_threads = size;



    int h, start, end, max_d;

    h = ceil(float(N)/float(num_threads));
  
    start = 1;
    end = h;
    if(end*(d+1) > N){
      end = (N % (h));
    }
  
    max_d = ceil((float)N/h)-1;
   
  
    size_t size_ = (N*ceil(float(N)/float(num_threads))+N*2)*sizeof(double);
    printf("d: %d, start: %d, end: %d, h: %d, d = %d, num_threads = %d, max_d = %d\n\n",d, start, end, h, d, num_threads, max_d);
  
  
    // usleep(d*40000);
  
    
    if(max_d >= d){
  
    cudaMallocHost( &a, size_ ); 
  
    cudaMalloc( &da,  size_ );
    
    cudaMallocHost( &newa,  size_ ); 
  
    cudaMalloc(&ddiff,  size_ );
  
    cudaMalloc(&dnewa,  size_ );
  
    ddiff_ = (double*)malloc( size_ );
    
    if ( d == 0 ) {
      a[start*N]=10.0;
      a[(start+1)*N-1]=20;  
      newa[start*N]=10.0;
      newa[(start+1)*N-1]=20;
    }
  
    if(d == max_d){
      a[(end)*N]=20.0;
      a[(end)*N + N-1]=30;
      newa[(end)*N]=20.0;
      newa[(end)*N + N-1]=30;
    }
    
    double  d1=(10.0)/(N-1);
          
  
    for (int i = 1; i < N; i++){
      if(d==0){
        a[start*N + i]=10 + d1*i;
        newa[start*N + i]=10 + d1*i;
      }
      if(d == max_d){
        a[(end)*N + i]=20 + d1*i;
        newa[(end)*N + i]=20 + d1*i;
      }
      
      if(i>=h*d && i< h*d + end){
  
        a[(start+i - h*d)*N]=10 + d1*i;
  
        a[(start+i - h*d)*N+N-1]=20 + d1*i;
  
        
       
  
        newa[(start+i - h*d)*N]=10 + d1*i;
  
        newa[(start+i - h*d)*N+N-1]=20 + d1*i;
  
      }
    }
  
      MPI_Status* status;
      
      if (max_d != 0)
      {
      if(d == 0){
         MPI_Sendrecv(&a[end*N], N, MPI_DOUBLE, d+1, d,
                     &a[(end+1)*N], N, MPI_DOUBLE, d+1, d+1,
                         MPI_COMM_WORLD, status);
       }
     
     if((d>0)&&(d<max_d)){
         MPI_Sendrecv(&a[(end)*N], N, MPI_DOUBLE, d+1, d,
                       &a[N*(start - 1)], N, MPI_DOUBLE, d-1, d-1,
                         MPI_COMM_WORLD, status);
     
         MPI_Sendrecv(&a[start*N], N, MPI_DOUBLE, d-1 , d,
                       &a[(end+1)*N], N, MPI_DOUBLE, d+1, d+1,
                         MPI_COMM_WORLD, status);
       }
     
       if(d == max_d){
         MPI_Sendrecv(&a[start*N], N, MPI_DOUBLE, d-1 , d,
           &a[(start-1)*N], N, MPI_DOUBLE, d-1, d-1,
             MPI_COMM_WORLD, status);
       }
      }
  
    //   MPI_Comm comm;
    // MPI_Barrier(comm);
    usleep((d+1) * 500000 );
    // printf("tuta");
  
  
  
    /////////////////////////////////////////////////////////////////////////////////////// PRINT
    for (int i = start-1; i <= end + 1; i ++){
      printf("\n");
      for (int j = 0; j < N; j ++){
        if (j==0)
          if((i<start)||(i>end))
            printf("-%d ", h*d + i);
          else
            printf("+%d ", h*d + i);
        printf("%f ", a[(i)*N + j]);
        }
      }
      printf("\n"); 
    ////////////////////////////////////////////////////////////////////////////////////// END PRINT
  
    // cudaStream_t stream1, stream2, stream3, stream4;
    // cudaStreamCreate ( &stream1);
    // cudaStreamCreate ( &stream2);
    
    
    // cudaMemcpyAsync( da, a, size_, cudaMemcpyHostToDevice, stream1);
    // cudaMemcpyAsync( dnewa, newa, size_, cudaMemcpyHostToDevice, stream2);
    
    // cudaStreamSynchronize(stream1);
    // cudaStreamSynchronize(stream2);
    // cudaStreamCreate ( &stream4);
    
  
  
  
  
  
   }











    cudaDeviceSynchronize();
   
    /* Shutdown MPI library */
    MPI_Finalize();

    return 0;
}
// ranks % 2 = 0 | ranks = 1 