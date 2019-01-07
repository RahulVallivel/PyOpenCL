import time
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import pyopencl as cl
import pyopencl.array
import numpy as np


class Transpose:
    def __init__(self,a_cpu):
        NAME='NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name==NAME:
                devs = platform.get_devices()
        
        # TODO:
        # Set up a command queue:
        self.ctx = cl.Context(devs) #setting up context
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE) #setting up command queue
        # host variables
        self.a_cpu = a_cpu  #copying the input data to variable
        self.b_cpu=np.zeros([self.a_cpu.shape[1],self.a_cpu.shape[0]])
    
        # device memory allocation
        self.a_gpu = cl.array.to_device(self.queue, self.a_cpu) #copy a to device

        self.b_gpu = cl.array.empty(self.queue, tuple([self.a_cpu.shape[1],self.a_cpu.shape[0]]), self.a_cpu.dtype) #create an empty array in device for the result
        # kernel code
        self.kernel_code = """
        __kernel void transpose(__global float *a_t, __global float *a, unsigned a_width, unsigned a_height)
          {
          int read_idx = get_global_id(0) + get_global_id(1) * a_width;
          int write_idx = get_global_id(1) + get_global_id(0) * a_height;
          a_t[write_idx] = a[read_idx];
          }
        """ 
    def serial_transpose(self):
        start = time.time()
        for i in range(self.a_cpu.shape[1]):
            for j in range(self.a_cpu.shape[0]):
                self.b_cpu[i,j]=self.a_cpu[j,i]

        self.times_cpu=time.time()-start
        return self.b_cpu,self.times_cpu
            
    
    def parallel_transpose(self):
        # return: an array containing capitalized characters from idata and running time.
        # TODO:
        # function call
        prg = cl.Program(self.ctx, self.kernel_code).build() #compiling the kernel
        start = time.time()
            
        prg.transpose(self.queue, tuple([self.a_cpu.shape[1],self.a_cpu.shape[0]]), None, self.b_gpu.data, self.a_gpu.data, np.uint32(self.a_cpu.shape[1]), np.uint32(self.a_cpu.shape[0]) )#kernel call
        self.times_gpu=time.time()-start
        # memory copy to host
        b = self.b_gpu.get() #copy result to host variable
        # Return output and measured time
        return b,self.times_gpu

class MatrixMultiply:
    def __init__(self, a_cpu, b_cpu):
        NAME='NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name==NAME:
                devs = platform.get_devices()

        self.ctx = cl.Context(devs) #setting up context
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE) #setting up command queue
        self.a_cpu=a_cpu 
        self.b_cpu=b_cpu
        self.a_gpu = cl.array.to_device(self.queue, self.a_cpu)
        self.b_gpu = cl.array.to_device(self.queue, self.b_cpu)
        self.c_gpu = cl.array.empty(self.queue, tuple([self.a_cpu.shape[0],self.b_cpu.shape[1]]), self.a_cpu.dtype) 
        self.TILE_WIDTH=self.a_cpu.shape[0]
        
        self.kernel = """
        __kernel void MatrixMulKernel_naive(const int M, const int N, const int K,__global float *A,__global float *B,__global float *C)
        {
   // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
 
    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<N; k++) {
        acc += A[k+ N * globalRow] * B[globalCol+ K * k];
        

    }
 
    // Store the result
    C[globalCol*K + globalRow] = acc;
}

#define BLOCK_SIZE %(TILE_DIM)s
#define SIMD_WORK_ITEMS 4 
__kernel
__attribute((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
 void MatrixMulKernel_optimized1(const int M, const int N, const int K,__global float* A,__global float* B,__global float* C)
   
{   const int A_width = N;
    const int B_width = K;
   
    __local float A_local[BLOCK_SIZE][BLOCK_SIZE];
    __local float B_local[BLOCK_SIZE][BLOCK_SIZE];

    
    int block_x = get_group_id(0);
    int block_y = get_group_id(1);

   
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

   
    int a_start = A_width * BLOCK_SIZE * block_y;
    int a_end   = a_start + A_width - 1;
    int b_start = BLOCK_SIZE * block_x;

    float running_sum = 0.0f;

   
    for (int a = a_start, b = b_start; a <= a_end; a += BLOCK_SIZE, b += (BLOCK_SIZE * B_width))
    {
        A_local[local_y][local_x] = A[a + A_width * local_y + local_x];
        B_local[local_x][local_y] = B[b + B_width * local_y + local_x];
    
        
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            running_sum += A_local[local_y][k] * B_local[local_x][k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }



    // Store result in matrix C
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = running_sum;
}

#define BLOCK_SIZE %(TILE_DIM)s
#define SIMD_WORK_ITEMS 4 
__kernel
__attribute((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
 void MatrixMulKernel_optimized2(const int M, const int N, const int K,__global float* A,__global float* B,__global float* C)
   
{   const int A_width = N;
    const int B_width = K;
    
    __local float A_local[BLOCK_SIZE][BLOCK_SIZE];
    __local float B_local[BLOCK_SIZE][BLOCK_SIZE+1];

    int block_x = get_group_id(0);
    int block_y = get_group_id(1);

  
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

   
    int a_start = A_width * BLOCK_SIZE * block_y;
    int a_end   = a_start + A_width - 1;
    int b_start = BLOCK_SIZE * block_x;

    float running_sum = 0.0f;

   
    for (int a = a_start, b = b_start; a <= a_end; a += BLOCK_SIZE, b += (BLOCK_SIZE * B_width))
    {
        
        A_local[local_y][local_x] = A[a + A_width * local_y + local_x];
        B_local[local_x][local_y] = B[b + B_width * local_y + local_x];
    
        barrier(CLK_LOCAL_MEM_FENCE);

       
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            running_sum += A_local[local_y][k] * B_local[local_x][k];
        }

        
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = running_sum;
}

    """

    def matrix_mul_naive(self):
        TILE_DIM = self.TILE_WIDTH
        self.kernel_1 = self.kernel % {
        'TILE_DIM': TILE_DIM}

        prg = cl.Program(self.ctx, self.kernel_1).build()
        start = time.time()
        prg.MatrixMulKernel_naive(self.queue,tuple([self.a_cpu.shape[0],self.b_cpu.shape[1]]), None,np.int32(self.a_cpu.shape[0]),np.int32(self.a_cpu.shape[1]),np.int32(self.b_cpu.shape[1]),self.a_gpu.data, self.b_gpu.data,self.c_gpu.data)
        times_gpu_=time.time()-start
        c_cpu= self.c_gpu.get()
        return c_cpu,times_gpu_



    def matrix_mul_optimized1(self):
        TILE_DIM = self.TILE_WIDTH
        self.kernel_1 = self.kernel % {
        'TILE_DIM': TILE_DIM}
        
        prg = cl.Program(self.ctx, self.kernel_1).build()
        start = time.time()
        prg.MatrixMulKernel_optimized1(self.queue,tuple([self.a_cpu.shape[0],self.b_cpu.shape[1]]), (TILE_DIM,TILE_DIM),np.int32(self.a_cpu.shape[0]),np.int32(self.a_cpu.shape[1]),np.int32(self.b_cpu.shape[1]),self.a_gpu.data, self.b_gpu.data,self.c_gpu.data)
        times_gpu_=time.time()-start
        c_cpu= self.c_gpu.get()
        return c_cpu,times_gpu_

    def matrix_mul_optimized2(self):
        TILE_DIM = self.TILE_WIDTH
        self.kernel_1 = self.kernel % {
        'TILE_DIM': TILE_DIM}
        
        prg = cl.Program(self.ctx, self.kernel_1).build()
        start = time.time()
        prg.MatrixMulKernel_optimized1(self.queue,tuple([self.a_cpu.shape[0],self.b_cpu.shape[1]]), (TILE_DIM,TILE_DIM),np.int32(self.a_cpu.shape[0]),np.int32(self.a_cpu.shape[1]),np.int32(self.b_cpu.shape[1]),self.a_gpu.data, self.b_gpu.data,self.c_gpu.data)
        times_gpu_=time.time()-start
        c_cpu= self.c_gpu.get()
        return c_cpu,times_gpu_


def main():
    

    a_cpu=np.random.randn(5,4).astype(np.float32)
    b_cpu=a_cpu.T.copy()
  
    mul=MatrixMultiply(a_cpu,b_cpu)
        
    c_cpu1,times_gpu_1=mul.matrix_mul_optimized1()
 
    
    print(c_cpu1)
    #print(c_cpu2)
    #print(c_cpu3)
    print(np.dot(a_cpu,b_cpu))
    print(np.sum(c_cpu1-np.dot(a_cpu,b_cpu)))
    #print(np.sum(c_cpu2-np.dot(a_cpu,b_cpu)))
    #print(np.sum(c_cpu3-np.dot(a_cpu,b_cpu)))
       
if __name__ == '__main__':
    main()