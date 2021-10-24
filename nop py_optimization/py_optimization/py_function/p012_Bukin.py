import numpy as np

def p012_Bukin(x):
    x1 = x[0]
    x2 = np.power(x1, 2)
    x3 = x[1]
    o = 100 * (np.abs(x3 - 0.01 * x2)) + 0.01 * (np.abs(x1 + 10))**2
    return (o)