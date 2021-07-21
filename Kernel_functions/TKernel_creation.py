import numpy as np
from PMKLpy import KernelFunctions
x = np.random.rand(100, 2)
q = 1
Lower, Upper =  x.min(axis = 0) - 0.1 , x.max(axis = 0) + 0.1
Kernel = KernelFunctions.Kernel(x, Lower, Upper,  q)
### Special Kernel structure for fast computing
P = np.eye(6) ## Just a some matrix of shape (2q, 2q)
TK = KernelFunctions.makeK(Kernel, P) 
### ndarray TKernel of shape (n_samples, n_samples)
