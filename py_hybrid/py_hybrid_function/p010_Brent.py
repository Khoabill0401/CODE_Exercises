import numpy as np

def p010_Brent (x):
    x = x.flatten()
    x1 = x[0]
    x2 = x[1]
    fx = (x1 + 10)**2 + (x2 + 10)**2 + np.exp(-x1**2 - x2**2)
    return (fx)