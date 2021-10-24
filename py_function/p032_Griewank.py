import numpy as np

def p032_Griewank(xx):
    dim = len (xx)
    sum = 0
    prod = 1
    for ii in range(dim):
        xi = xx[ii]
        sum += (xi ** 2) / 4000
        prod *= np.cos(xi / np.sqrt([ii + 1]))
        fx = sum - prod + 1
    return fx