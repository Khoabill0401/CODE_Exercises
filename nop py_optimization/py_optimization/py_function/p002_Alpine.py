import numpy as np
import math
def p002_Alpine (x):
    dim = len (x)
    x = x.flatten()
    sumx=0
    for i in range (dim):
        sumx += abs(x[i]*math.sin(x[i]) + 0.1 *x[i])
    return (sumx)