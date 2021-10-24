import numpy as np
import math

def p049_Paviani (x):
    dim = 9
    sum = 0
    prod = 1
    for i in range(dim):
        a = np.log(x[i]-2)**2 + np.log(10-x[i])**2
        sum += a
        prod = prod * pow(x[i],0.2)
        fx = sum - prod
    return (fx)

