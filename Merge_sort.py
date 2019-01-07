
import pyopencl as cl
import pyopencl.array
import time
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
import random
from array import array
class MergeSort:
    def merge(self,a, left, mid, right):
    
        copy_list = []
        i, j = left, mid + 1
        ind = left
        
        while ind < right+1:
            
           
            if i > mid:
                copy_list.append(a[j])
                j +=1
           
            elif j > right:
                copy_list.append(a[i])
                i +=1
           
            elif a[j] < a[i]:
                copy_list.append(a[j])
                j +=1
            else:
                copy_list.append(a[i])
                i +=1
            ind +=1
            
        ind=0
        for x in (xrange(left,right+1)):
            a[x] = copy_list[ind]
            ind += 1
    

    def merge_sort_serial(self,list_):
    
        factor = 2
        temp_mid = 0
        
        while 1:
            index = 0
            left = 0
            right = len(list_) - (len(list_) % factor) - 1
            mid = (factor / 2) - 1
            
           
            while index < right:
                temp_left = index
                temp_right = temp_left + factor -1
                mid2 = (temp_right +temp_left) / 2
                self.merge (list_, temp_left, mid2, temp_right)
                index = (index + factor)
            
        
          
            if len(list_) % factor and temp_mid !=0:
                
                self.merge(list_, right +1, temp_mid, len(list_)-1)
                
                mid = right
            
            factor = factor * 2
            temp_mid = right
           
            
            if factor > len(list_) :
                mid = right
                right = len(list_)-1
                self.merge(list_, 0, mid, right)
                break
        return list_


    def merge_sort_naive(self, a):

        NAME='NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name==NAME:
                devs = platform.get_devices()
        
        a_cpu = np.array(a)
        if(a_cpu.shape[0]==1):
            return a_cpu.to_list()
        
        ctx = cl.Context(devs)
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
    
        print(a_cpu)
        
        a_gpu = cl.array.to_device(queue,(a_cpu.astype(np.int32)))

        len_gpu=cl.array.to_device(queue,np.array(a_cpu.shape[0]))
    
        b_gpu =cl.array.empty(queue,a_cpu.shape, a_cpu.dtype)

        kernel_code = """
       
        __kernel void merge(__global const int *arr1,  int arr1size, __global const int *arr2, int arr2size, __global int *out)
        {
        
        int x = 0;
        int y = 0;
        int n = 0;
        while(x < arr1size || y < arr2size) 
        {
        if(x < arr1size && y < arr2size)
       {
        if(arr1[x] < arr2[y]) 
        {
          out[n] = arr1[x];
          n++;
          x++;
        }
        
         else 
        {
          out[n] = arr2[y];
          n++;
          y++;
        }
      } 
    
      else if(x < arr1size) 
      {
        out[n] = arr1[x];
        n++;
        x++;
      } 
     
      else if(y < arr2size) 
      {
        out[n] = arr2[y];
        n++;
        y++;
      }
    
    }
}



__kernel void cpy(__global int *out,__global const int *in, int size) 
{
    int x = 0;

    for(x = 0; x < size; x++) 
    {
      out[x] = in[x];
      barrier(CLK_LOCAL_MEM_FENCE);
      

      
      
      }

    
  }

  


  __kernel void mergesort_naive(__global int *a, __global int *b, __global int *len)
  {     

        
    int size = 1;
    int indx = get_global_id(0) * 2;
    int i = 0;

    while(size <=*len) 
    {
      barrier(CLK_LOCAL_MEM_FENCE);


      if((indx + size) <*len) 
      {
        i = (indx + size) + (((indx + (size * 2)) > *len) ? *len - (indx + size) : size);
        
        
     
        merge(&a[indx], size, &a[indx + size], ((indx + (size * 2)) > *len) ? *len - (indx + size) : size, &b[indx]);
       
      
        
      } 
      else 
      {
       

        return;

      }
      size *= 2;
      
      
     
      if(size>*len)
      {
        
        break;
      }
      
      cpy(&a[indx], &b[indx], size);
      
      if(indx%(2*size) != 0) 
      {
        
        return;
        
      }
    }
    *len = i;
    barrier(CLK_LOCAL_MEM_FENCE);
    

  }
        """


     
        for i in range(1,33):
            if (a_cpu.shape[0]/(i)<=1024):
                block_threads=int(np.floor(a_cpu.shape[0]/(i))+1)
                grid_blocks=int(i)
                break
        

        prg = cl.Program(ctx,kernel_code).build()
        start = time.time()
        prg.mergesort_naive(queue,(a_cpu.shape[0]/2,1),None,a_gpu.data, b_gpu.data, len_gpu.data)        
        times_cpu=time.time()-start
        b_cpu=b_gpu.get()

        return b_cpu.tolist(),times_cpu


    def merge_sort_optimized1(self, a_cpu):
        NAME='NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name==NAME:
                devs = platform.get_devices()

        ctx = cl.Context(devs) #setting up context
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
        a_cpu = np.int32(np.array(a_cpu).copy())
        if(a_cpu.shape[0]==1):
            return a_cpu.to_list()
        length = np.int32(np.array(a_cpu.shape[0]))
        result = np.int32(np.zeros(length))

        a_gpu = cl.array.to_device(queue,np.array(a_cpu))
        len_gpu=cl.array.to_device(queue,np.array(length))
        b_gpu =cl.array.empty(queue,a_cpu.shape[0], a_cpu.dtype)

        kernel_code_opti = """

        #define BLOCK_SIZE 1024

        __kernel void merge(__local const int *arr1, int arr1size, __local const int *arr2, int arr2size, __global int *out)
        {
            int x = 0;
            int y = 0;
            int n = 0;


            while(x < arr1size || y < arr2size){
                if(x < arr1size && y < arr2size){
                    if(arr1[x] < arr2[y]){
                        out[n] = arr1[x];
                        n++;
                        x++;
                    }
                    else{
                        out[n] = arr2[y];
                        n++;
                        y++;

                    }
                }
                else if(x < arr1size){
                    out[n] = arr1[x];
                    n++;
                    x++;
                }
                else if(y < arr2size){
                    out[n] = arr2[y];
                    n++;
                    y++;
                }

            }

        }

        __kernel void cpy(__local int *out, __global const int *in, int size){
            int x = 0;
            for(x = 0; x < size; x++){
                out[x] = in[x];
            
            }
        }

        __kernel void mergesort_optimized(__global int *a, __global int *b, __global int *len){
            __local int a_local[BLOCK_SIZE];

            int size = 1;
            int indx = get_global_id(0) * 2;
            int i = 0;

            for(int k = 0; k < *len; k++){
                a_local[k] = a[k];
            }

            while(size <= *len){
                barrier(CLK_LOCAL_MEM_FENCE);
                if((indx + size) < *len){
                    i = (indx + size) + (((indx + (size * 2)) > *len) ? *len - (indx + size) : size);
                    merge(&a_local[indx], size, &a_local[indx + size], ((indx + (size * 2)) > *len) ? *len - (indx + size) : size, &b[indx]);
                }
                else{
                    return;
                }
                size *= 2;
                if(size > *len){
                    break;
                }
                cpy(&a_local[indx], &b[indx], size);
                if(indx%(2 * size) != 0){
                    return;
                }
            }
            *len = i;
        }
        """


        prg = cl.Program(ctx,kernel_code_opti).build()
        start = time.time()
        prg.mergesort_optimized(queue,(866,1),None,a_gpu.data, b_gpu.data, len_gpu.data)        
        times_cpu=time.time()-start
        b_cpu=b_gpu.get()

        return b_cpu.tolist(),times_cpu



def main():
    ti_c=[]
    ti_gn=[]
    ti_go=[]
    
    for i in range(1,2):
        sort= MergeSort()
        a_cpu = np.random.randint(1,9000,size=1024).astype(np.int32)
        start = time.time()
        sorted_a_serial = sort.merge_sort_serial(a_cpu.copy())
        ti_serial =time.time()-start
        sorted_a_naive,ti_naive = sort.merge_sort_naive(a_cpu.copy())
        sorted_a_optimized,ti_opti = sort.merge_sort_optimized1(a_cpu.copy())
        del sort
        ti_c.append(ti_serial)
        ti_gn.append(ti_naive)
        ti_go.append(ti_opti)
    sorted_a_cpu=sorted_a_serial
    sorted_a_gpu_naive=sorted_a_naive
    sorted_a_gpu_optimized=sorted_a_optimized
    plt.plot(ti_c)
    plt.plot(ti_gn)
    plt.plot(ti_go)

    plt.xlabel('size of input')
    plt.ylabel('time')
    plt.title('python vs pyopencl')
    plt.legend(['cpu','gpu_naive','gpu_optimized'])
    plt.savefig('pyopencl_only_kernel.png')
    plt.close()

    print("CPU:",sorted_a_cpu)
    print("GPU_naive:",sorted_a_gpu_naive)
    print("GPU_optimized:",sorted_a_gpu_optimized)

    print("is Naive code correct:",sorted_a_cpu==sorted_a_gpu_naive)
    print("is optiomized code correct:",sorted_a_cpu==sorted_a_gpu_optimized)       

if __name__=='__main__':
    main()
