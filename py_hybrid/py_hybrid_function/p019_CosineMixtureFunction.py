import numpy as np

def p019_CosineMixtureFunction(x):
    dim = len(x)
    x = x.flatten()
    xsum = 0
    xsumcos = 0
    for i in range(dim):
        xsum += x[i]**2
        xsumcos += np.cos(5*np.pi*x[i])
    o = -0.1*xsumcos + xsum
    return (o)

