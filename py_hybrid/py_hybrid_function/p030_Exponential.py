import numpy as np

def p030_Exponential (x):
    dim = len (x)
    x = x.flatten()
    sumx=0
    for i in range (dim):
        sumx += x[i]**2
        fx = -np.exp(-0.5*sumx)
        return (fx)