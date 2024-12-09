import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
print(pycuda.VERSION)
X_host=np.array([1,2,3],dtype=np.float32)
Y_host=np.array([1,1,1],dtype=np.float32)
Z_host=np.array([2,2,2],dtype=np.float32)


x_device=gpuarray.to_gpu(X_host)
y_device=gpuarray.to_gpu(Y_host)
z_device=gpuarray.to_gpu(Z_host)
lol=X_host+Y_host
print("the x and y in the normal memeory are: ",lol)
devdata=(x_device+y_device).get()
print("the gpu array addings are: ",devdata)
