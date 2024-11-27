import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
print(pycuda.VERSION)
host_data=np.array([1,2,3,4,5],dtype=np.float32)
device_data=gpuarray.to_gpu(host_data)
device_datax2=device_data * 2
# print(device_data)
# print(device_datax2)
host_datax2=device_datax2.get()
print(host_datax2)