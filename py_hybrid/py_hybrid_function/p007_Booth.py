import numpy as np

def p007_Booth(x):
    x = x.flatten()
    x1 = x[0]
    x2 = x[1]
    fx= (x1 + 2*x2 -7)**2 + (2*x1 + x2 -5)**2
    return (fx)