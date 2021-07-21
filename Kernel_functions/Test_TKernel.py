from PMKLpy import Transformation
from PMKLpy import KernelFunctions
import numpy as np

x = np.random.rand(200, 2)
x1 = x[:100]
x2 = x[100:]
Lower, Upper =  x.min(axis = 0) - 0.1 , x.max(axis = 0) + 0.1
P = np.eye(6)
Z1 = Transformation.monomials(x1, 1)
Z2 = Transformation.monomials(x2, 1)

TK = KernelFunctions.TKtest(x1,x2,Z1,Z2,Lower, Upper,P)