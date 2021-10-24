import math
import numpy as np
from scipy.optimize import differential_evolution
from numpy.random import rand
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi

# objective function
def objective(x):
    x = x.flatten()
    dim = 20
    xsum = 0
    xsumcos = 0
    for i in range(dim):
        xsum += x[i] ** 2
        xsumcos += np.cos(2 * np.pi * x[i])
    o = -20 * np.exp(-0.2 * np.sqrt(xsum / dim)) - np.exp(xsumcos / dim) + 20 + np.e
    return (o)

dim = 20
# define range for input
r_min, r_max = -35, 35
# define the bounds on the search
bounds = []
for i in range(dim):
    bounds.append([r_min, r_max])
# perform the differential evolution search
result = differential_evolution(objective, bounds)
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))