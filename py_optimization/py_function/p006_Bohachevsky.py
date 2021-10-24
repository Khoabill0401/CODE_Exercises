import numpy as np

def p006_Bohachevsky(x):
    x = x.flatten()
    x1, x2 = x
    fx = x1**2 + 2 * x2**2 - 0.3 * np.cos(3 * np.pi * x1) - 0.4 * np.cos(4 * np.pi * x2) + 0.7
    return (fx)