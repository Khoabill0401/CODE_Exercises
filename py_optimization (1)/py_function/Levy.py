import math
import numpy as np
import random
from scipy.special import gamma, factorial

def Levy(d):
    beta = 3/2
    sigma = (gamma(1 + beta)*math.sin(math.pi*beta/3)/(gamma((1 + beta)/2)*beta*2**((beta - 1)/2)**(1/beta)))
    u = np.random.randn(1, d)*sigma
    v = np.random.randn(1, d)
    step = u/np.abs(v)**(1/beta)
    L = 0.01*step
    return (L)