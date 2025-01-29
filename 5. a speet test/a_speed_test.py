import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time
host_data=np.float32(np.random.random(50000000))
t1=time()
host_data_2x=host_data*np.float32(2)
t2=time()
print("total time to compute on cpu: %f" %(t2-t1))
device_data=gpuarray.to_gpu(host_data)

t1=time()
device_data_2x=device_data*np.float32(2)
t2=time()
from_device=device_data_2x.get()
print("total time to compute on Gpu: %f" %(t2-t1))

print("is the host computation the same as the GPU computation? {}".format(np.allclose(from_device,host_data_2x)))



""" codes to run on ipython
with open('a_speed_test.py','r') as f:
    timecalcode=f.read()
%prun -s cumulative exec(timecalcode)"""
# test
# test