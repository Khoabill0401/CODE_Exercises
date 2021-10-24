import numpy as np

def p045_Mishra(x):
    dim = len (x)
    sum=0
    for i in range (dim):
        sum += x[i]
    fx = (10000 * abs(sum))**0.5
    return (fx)