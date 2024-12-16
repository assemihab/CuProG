import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
import numpy as np
import matplotlib.pyplot as plt
from time import time

mandel_ker=ElementwiseKernel(
    "pycuda::complex<float> *lattice,float *mandelbrotgraph,int max_iter,float upper_bound",
    """
    mandelbrotgraph[i]=1;
    pycuda::complex<float> c=lattice[i];
    pycuda:: complex<float> z(0,0);
    for (int j =0; j<max_iter;j++)
    {
    z=z*z+c;
    if (abs(z)>upper_bound)
        {
        mandelbrotgraph[i]=0;
        break;
        }
    }
    """,
    'mandel_ker'
)

def mandelBrotLattice(width,height,xmin,xmax,ymin,ymax,max_iter,upper_bound):
    real_vals = np.matrix(np.linspace(xmin, xmax, width), dtype=np.complex64)
    imag_vals = np.matrix(np.linspace(ymin, ymax, height), dtype=np.complex64) * 1j
    mandelbrotLatticee=np.array(real_vals + imag_vals.transpose(), dtype=np.complex64)
    mandelbrotLatticee_gpu=gpuarray.to_gpu(mandelbrotLatticee)
    mandelbrotgraph=gpuarray.empty(shape=mandelbrotLatticee.shape, dtype=np.float32)
    mandel_ker(mandelbrotLatticee_gpu,mandelbrotgraph,np.int32(max_iter),np.float32(upper_bound))
    return mandelbrotgraph

if __name__ == "__main__":
    t1 = time()
    mandelbrotgraph=mandelBrotLattice(512,512,-2,2,-2,2,520, 100)
    t2 = time()
    mandeltime=t2-t1


    mandelbrotgraphh=mandelbrotgraph.get()
    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandelbrotgraphh, extent=(-2, 2, -2, 2))
    plt.savefig('mandelbrot.png', dpi=fig.dpi)
    t2 = time()

    dumptime = t2-t1
    print("Time for mandelbrot: ", mandeltime)
    print("Time for dumping the image: ", dumptime)


