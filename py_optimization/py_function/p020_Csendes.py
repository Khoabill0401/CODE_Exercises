import numpy as np

def p020_Csendes (x):
    dim = len(x)
    x = x.flatten()
    sumx = 0
    for i in range(dim):
        sumx += (x[i]**6)*(2 + np.sin(1/x[i]))
    return sumx