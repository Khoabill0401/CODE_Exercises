import numpy as np

def p003_BartelsConn(x):
    x=x.flatten()
    x1, x2 = x
    fx = abs(x1**2.0 + x2**2.0 +x1*x2) + abs(np.sin(x1)) + abs(np.cos(x2))
    return fx