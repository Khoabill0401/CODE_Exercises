import numpy as np

def p077_Stepint(x):
    dim = len(x)
    x = x.flatten()
    xsum = 0
    for i in range(dim):
        xsum += np.ceil(x[i])
    o = 25 + xsum
    return (o)

