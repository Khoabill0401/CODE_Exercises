import math
import numpy as np
def p065_Salomon(x):
    dim = len(x)
    x = x.flatten()
    xsum2 = 0
    for i in range( dim ):
        xsum2 +=  x[i]**2
    o = 1 - np.cos(2*np.pi**2*np.sqrt(xsum2))+0.1*np.sqrt(xsum2)
    return (o)
