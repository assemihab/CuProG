import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time
from pycuda.elementwise import ElementwiseKernel
hostData=np.float32(np.random.random(50000000))
gpu2xker=ElementwiseKernel("float *in, float*out",
                           "out[i]=2*in[i];",
                           "gpu2xker")
def speedCompar():
    t1=time()
    hostData2x=hostData*np.float32(2)
    t2=time()
    print("the total time for the CPU is: %f" %(t2-t1))
    deviceData=gpuarray.to_gpu(hostData)
    deviceData2x=gpuarray.empty_like(deviceData)
    t1=time()
    gpu2xker(deviceData,deviceData2x)
    t2=time()
    fromDevice=deviceData2x.get()
    print("the total time for the GPU is: %f" %(t2-t1))

    print("is the host same as GPU?: {}".format(np.allclose(fromDevice,hostData2x)))
if __name__=='__main__':
    speedCompar()
