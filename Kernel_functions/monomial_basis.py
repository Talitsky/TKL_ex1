import numpy as np
from PMKLpy import Transformation
x = np.random.rand(100, 2)
Z = Transformation.monomials(x, 1)