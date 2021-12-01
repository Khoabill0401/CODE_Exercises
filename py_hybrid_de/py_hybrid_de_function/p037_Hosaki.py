import numpy as np

def p037_Hosaki(x):
    x1 = x[0]
    x2 = x[1]
    o = (1 - 8*x1 + 7*(x1**2) -(7/3)*(x1**3) + (1/4)*(x1**4))*(x2**2)*(np.exp(-x2))
    return (o)