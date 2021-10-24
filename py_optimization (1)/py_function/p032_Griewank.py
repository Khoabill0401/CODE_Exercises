import numpy as np

def p032_Griewank(x):
    dim = len (x)
    x = x.flatten()
    sum = 0
    prod = 1
    for i in range(dim):
        sum += (x[i] ** 2) / 4000
        prod *= np.cos(x[i] / np.sqrt([i + 1]))
        fx = sum - prod + 1
    return fx