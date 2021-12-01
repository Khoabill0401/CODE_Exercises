import math
import numpy as np
def p071_Schumer(x):
    dim = len(x)
    x = x.flatten()
    xsum4 = 0
    for i in range(dim):
        xsum4 += x[i]**4
    o = xsum4
    return (o)
