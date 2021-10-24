import numpy as np
from numpy import sin
from numpy import cos
from numpy import exp
def p005_Bird (x):
    x = x.flatten ()
    x1, x2 = x
    a = (1-np.cos(x2))**2
    b = (1-np.sin(x1))**2
    fx = np.sin(x1)*np.exp(a) + np.cos(x2)*np.exp(b) + (x1 - x2)**2
    return (fx)