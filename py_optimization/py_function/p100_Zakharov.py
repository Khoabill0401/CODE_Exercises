import numpy as np

def p100_Zakharov(x):
    dim = len(x)
    x = x.flatten()
    xsum2 = 0
    xsumi = 0
    for i in range(dim):
        xsum2 += x[i]**2
        xsumi += i*x[i]
    o =  xsum2 + (0.5*xsumi)**2 + (0.5*xsumi)**4
    return(o)
