import numpy as np
def p058_Quintic(x):
    dim = len(x)
    x = x.flatten()
    xsum =0
    for i in range(dim):
        xsum += abs( x[i]**5 - 3*x[i]**4 + 4*x[i]**3 +2*x[i]**2-10*x[i] -4)
    o = xsum
    return (o)

