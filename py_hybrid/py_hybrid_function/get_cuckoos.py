"""
# ==================================================================================== #
# Get cuckoos by random walk                                                           #
# ==================================================================================== #
# This work has been done by:                                                          #
# Nguyen Anh Khoa  - 1810240                                                           #
# Phone: 0868.840.441                                                                  #
# Email: khoa.nguyen41@hcmut.edu.vn                                                    #
# ==================================================================================== #
"""
import math
import numpy as np
from scipy.special import gamma
from py_hybrid_function.simplebounds import *

def get_cuckoos(nest, best, Lb, Ub):
    # Levy flights
    n = len(nest)
    # Levy exponent and coefficient
    # For details, see equation (2.21), Page 16 (Chapter 2) of the book
    # X. S. Yang, Nature-Inspired Metaheuristic Algorithms, 2nd Edition, Luniver Press, (2010).

    beta = 3/2
    sigma = (gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
                gamma((1 + beta) / 2) * beta * (2 ** (((beta - 1) / 2))))) ** (1 / beta)

    for j in range(n):
        s = nest[j, :]
        # This is a simple way of implementing Levy flights
        # For standard random walks, use step=1;
        ## Levy flights by Mantegna's algorithm

        u = np.random.randn(len(s))*sigma
        v = np.random.randn(len(s))
        step = u/(np.power(abs(v), (1/beta)))

        # In the next equation, the difference factor (s-best) means that
        # when the solution is the best solution, it remains unchanged.
        stepsize = 0.001*step*(s - best)
        # Here the factor 0.01 comes from the fact that L/100 should the typical
        # step size of walks/flights where L is the typical lenghtscale;
        # otherwise, Levy flights may become too aggresive/efficient,
        # which makes new solutions (even) jump out side of the design domain
        # (and thus wasting evaluations).
        # Now the actual random walks or flights
        s += stepsize*np.random.randn(len(s))
        # Apply simple bounds/ limits
        nest[j, :] = simplebounds(s, Lb, Ub)

    return (nest)