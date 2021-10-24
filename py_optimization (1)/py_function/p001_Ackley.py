import numpy as np

def p001_Ackley(x):
    dim = len(x)
    x = x.flatten()
    xsum = 0
    xsumcos = 0
    for i in range(dim):
        xsum += x[i]**2
        xsumcos += np.cos(2*np.pi*x[i])
    o = -20*np.exp(-0.2*np.sqrt(xsum/dim)) - np.exp(xsumcos/dim) + 20 + np.e
    return (o)

