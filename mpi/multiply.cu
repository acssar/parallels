/* multiply.cu */
#include <cuda.h>
#include <cuda_runtime.h>
 
 __global__ void __multiply__ ()
 {
 }
 
 extern "C" void call_me_maybe()
{
     /* ... Load CPU data into GPU buffers  */
 
     __multiply__ <<< ...block configuration... >>> (x, y);
 
     /* ... Transfer data from GPU to CPU */
}
