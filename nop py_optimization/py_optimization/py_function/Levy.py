import math
import numpy as np
from scipy.special import gamma

def Levy(d):
    beta = 3/2
    sigma = (gamma(1 + beta)*math.sin(math.pi*beta/2)/(gamma((1 + beta)/2)*beta*(2**(((beta - 1)/2)))))**(1/beta)
    u = np.random.randn(1, d)*sigma
    u = -0.0933797729154532
    v = np.random.randn(1, d)
    v = 0.868215725991768
    step = u/(np.abs(v)**(1/beta))
    L = 0.01*step
    return (L)