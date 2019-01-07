import time
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import pyopencl as cl
import pyopencl.array
import numpy as np



class openclModule:
    def __init__(self, idata):
        # idata: an array of lowercase characters.
        # Get platform and device
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        
        # TODO:
        # Set up a command queue:
        self.ctx = cl.Context(devs) #setting up context
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE) #setting up command queue
        # host variables
        self.a = idata  #copying the input data to variable
        self.times_gpu = 0 #initialising variable to measure gpu time
        self.times_cpu = 0 #initialising variable to measure cpu time
        # device memory allocation
        self.a_gpu = cl.array.to_device(self.queue, self.a) #copy a to device

        self.b_gpu = cl.array.empty(self.queue, self.a.shape, self.a.dtype) #create an empty array in device for the result
        # kernel code
        self.kernel = """
        __kernel void func(__global char* a, __global char* b) {
        unsigned int i = get_global_id(0);
        b[i] = a[i]-32;
        }
        """

    def runAdd_parallel(self):
        # return: an array containing capitalized characters from idata and running time.
        # TODO:
        # function call
        prg = cl.Program(self.ctx, self.kernel).build() #compiling the kernel
        start = time.time() #start timer
        prg.func(self.queue, self.a.shape, None, self.a_gpu.data,self.b_gpu.data) #kernel call
        self.times_gpu = time.time()-start #get time
        # memory copy to host
        b = self.b_gpu.get() #copy result to host variable
        # Return output and measured time
        return b,self.times_gpu #return result and time

    def runAdd_serial(self):
        # return: an array containing capitalized characters from idata and running time.
        output=[]  #initialise variable for result
        
        for i in self.a:   
            k=ord(i)    #get ascii value of each character
            start = time.time() #start timer
            k=k-32   #subtract 32 from ascii value to capitalise
            self.times_cpu=self.times_cpu + time.time()-start #get time
            i=chr(k) #convert modified ascii value back to character
            output.append(i) #append the results
        
        return output,self.times_cpu #return the output and time

def main():
    
    times_cpu=[] #initialise variable to copy time
    times_gpu=[]
    test =0 #test variable to check if cpu time is greater than gpu time

    for itr in range(1, 100):
        idata = list("abcdefghijklmnopqrstuvwxyz"*itr) #extend the array(input data)
        idata=np.array(idata) #convert list to numpy array
    ##############################################################################################
    #   capitalize idata using your serial and parallel functions, record the running time here  #
        p=openclModule(idata) #create class object
        gpu_output,times_gpu_=p.runAdd_parallel() #call function for gpu execution
        cpu_output,times_cpu_=p.runAdd_serial() #call function for cpu execution
        times_gpu.append(times_gpu_) #append time
        times_cpu.append(times_cpu_)

    ##############################################################################################
        print 'py_output=\n', cpu_output # py_output is the output of your serial function
        print 'parallel_output=\n', gpu_output # parallel_output is the output of your parallel function
        print 'Code equality:\t', (cpu_output==gpu_output) #check if cpu_output and gpu_output are same
        print 'string_len=', len(idata), '\tpy_time: ', times_cpu[itr-1], '\tparallel_time: ', times_gpu[itr-1] # py_time is the running time of your serial function, parallel_time is the running time of your parallel function.
    
    for i in range(len(times_cpu)):
        if(times_cpu[i]>=times_gpu[i]):
            print('the L_CL value is:{}'.format(i+1)) #calculating the iteration at which the cpu executon time becomes greater than gpu execution time
            test=test+1
            break
    if(test==0):
        print('CPU execution time did not cross GPU execution time, Increase the number of iterations to observe the GPU perform better than the CPU')
    

    plt.plot(times_gpu)
    plt.plot(times_cpu)
    plt.xlabel('iterations')
    plt.ylabel('time(seconds)')
    plt.legend(['pyopencl_gpu','python_cpu'])
    plt.savefig('time_opencl.png')
    plt.close()

if __name__ == '__main__':
    main()




