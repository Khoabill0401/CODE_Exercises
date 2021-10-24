import numpy as np
from numpy import pi
def p023_Deb(x):
    dim=len (x)
    x = x.flatten()
    sumx=0
    for i in range(dim):
        sumx += (np.sin(5 * pi * x[i]))**6
        fx = -sumx/dim
        return fx