"""
#==================================================================================================#
# Differential Evolution (DE) demo with Appendix A: Test Function Benchmark                        #
# This work has been done by:                                                                      #
# 1. Nguyen Anh Khoa - 1810240                                                                     #
# 2. Phan Vuong Phu - 1710235                                                                      #
# 3. Trang Si Tan Khang - 1810215                                                                  #
#==================================================================================================#
"""
import math
import numpy as np
import random
from py_function.Levy import *
from py_function.p001_Ackley import *
from py_function.p002_Alpine import *
from py_function.p003_BartelsConn import *
from py_function.p005_Bird import *
from py_function.p006_Bohachevsky import *
from py_function.p007_Booth import *
from py_function.p010_Brent import *
from py_function.p011_Brown import *
from py_function.p012_Bukin import *
from py_function.p019_CosineMixture import *
from py_function.p020_Csendes import *
from py_function.p023_Deb import *
from py_function.p030_Exponential import *
from py_function.p032_Griewank import *
from py_function.p037_Hosaki import *
from py_function.p045_Mishra import *
from py_function.p054_PowellSum import *
from py_function.p057_Quartic import *
from py_function.p058_Quintic import *
from py_function.p065_Salomon import *
from py_function.p071_SchumerSteiglitz import *
from py_function.p075_Sphere import *
from py_function.p076_Step import *
from py_function.p077_Stepint import *
from py_function.p079_SumSquares import *
from py_function.p080_StyblinskiTang import *
from py_function.p096_XinSheYangSecond import *
from py_function.p098_XinSheYangFourth import *
from py_function.p100_Zakharov import *
from numpy import clip
from numpy import argmin
from numpy import min
from numpy.random import rand
from numpy.random import choice
from numpy import around
from numpy import asarray

# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


# define boundary check operation
def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound


# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = rand(dims)
    # generate trial vector by binomial crossover
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial


def DE(Function_name, pop_size, maxiter, F, cr):
    # Determine what benchmark function
    if Function_name == 'p001_Ackley':
        d = 2
        Fun = lambda x: p001_Ackley(x)
        # bounds = asarray([(-35, 35), (-35, 35)])
        Ub = 35
        Lb = -35
        # define the bounds on the search
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p002_Alpine':
        d = 30
        Fun = lambda x: p002_Alpine(x)
        Lb = -35
        Ub = 35
        ptype = 1
    if Function_name == 'p003_BartelsConn':
        d = 30
        Fun = lambda x: p003_BartelsConn(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p005_Bird':
        d = 30
        Fun = lambda x: p005_Bird(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p006_Bohachevsky':
        d = 30
        Fun = lambda x: p006_Bohachevsky(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p007_Booth':
        d = 30
        Fun = lambda x: p007_Booth(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p010_Brent':
        d = 30
        Fun = lambda x: p010_Brent(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p011_Brown':
        d = 30
        Fun = lambda x: p011_Brown(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p012_Bukin':
        d = 2
        Fun = lambda x: p012_Bukin(x)
        #Lb = np.array([-15, -3])
        #Ub = np.array([-5, 3])
        Lb1 = -15
        Lb2 = -3
        Ub1 = -5
        Ub2 = 3
        bounds = np.array([[Lb1, Ub1], [Lb2, Ub2]])
        #for i in range(d):
        #    bounds.append([[Lb1, Ub1], [Lb2, Ub2]])
        #bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p019_CosineMixture':
        d = 2
        Fun = lambda x: p019_CosineMixture(x)
        Lb = -1
        Ub = 1
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p020_Csendes':
        d = 30
        Fun = lambda x: p020_Csendes(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p023_Deb':
        d = 30
        Fun = lambda x: p023_Deb(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p030_Exponential':
        d = 30
        Fun = lambda x: p030_Exponential(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p032_Griewank':
        d = 30
        Fun = lambda x: p032_Griewank(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p037_Hosaki':
        d = 2
        Fun = lambda x: p037_Hosaki(x)
        #Lb = np.array([0, 0])
        #Ub = np.array([5, 6])
        Lb1 = 0
        Lb2 = 0
        Ub1 = 5
        Ub2 = 6
        bounds = np.array([[Lb1, Ub1], [Lb2, Ub2]])
        ptype = 1
    if Function_name == 'p045_Mishra':
        d = 30
        Fun = lambda x: p045_Mishra(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p054_PowellSum':
        d = 30
        Fun = lambda x: p054_PowellSum(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p057_Quartic':
        d = 30
        Fun = lambda x: p057_Quartic(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p058_Quintic':
        d = 30
        Fun = lambda x: p058_Quintic(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p065_Salomon':
        d = 30
        Fun = lambda x: p065_Salomon(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p071_SchumerSteiglitz':
        d = 30
        Fun = lambda x: p071_SchumerSteiglitz(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p075_Sphere':
        d = 30
        Fun = lambda x: p075_Sphere(x)
        Lb = 0
        Ub = 10
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 0
    if Function_name == 'p076_Step':
        d = 30
        Fun = lambda x: p076_Step(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p077_Stepint':
        d = 30
        Fun = lambda x: p077_Stepint(x)
        Lb = -5.12
        Ub = 5.12
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p079_SumSquares':
        d = 30
        Fun = lambda x: p079_SumSquares(x)
        Lb = -35
        Ub = 35
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p080_StyblinskiTang':
        d = 2
        Fun = lambda x: p080_StyblinskiTang(x)
        Lb = -5
        Ub = 5
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p096_XinSheYangSecond':
        d = 2
        Fun = lambda x: p096_XinSheYangSecond(x)
        Lb = -5
        Ub = 5
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p098_XinSheYangFourth':
        d = 2
        Fun = lambda x: p098_XinSheYangFourth(x)
        Lb = -5
        Ub = 5
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1
    if Function_name == 'p100_Zakharov':
        d = 2
        Fun = lambda x: p100_Zakharov(x)
        Lb = -5
        Ub = 5
        bounds = []
        for i in range(d):
            bounds.append([Lb, Ub])
        bounds = np.array(bounds)
        ptype = 1

    # initialise population of candidate solutions randomly within the specified bounds
    pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    # evaluate initial population of candidate solutions
    obj_all = [Fun(np.array(ind)) for ind in pop]
    # find the best performing vector of initial population
    best_vector = pop[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    # initialise list to store the objective function value at each iteration
    obj_iter = list()
    # run iterations of the algorithm
    for i in range(maxiter):
        # iterate over all candidate solutions
        for j in range(pop_size):
            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
            # perform mutation
            mutated = mutation([a, b, c], F)
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            trial = crossover(mutated, pop[j], len(bounds), cr)
            # compute objective function value for target vector
            obj_target = Fun(np.array(pop[j]))
            # compute objective function value for trial vector
            obj_trial = Fun(np.array(trial))
            # perform selection
            if obj_trial < obj_target:
                # replace the target vector with the trial vector
                pop[j] = trial
                # store the new objective function value
                obj_all[j] = obj_trial
        # find the best performing vector at each iteration
        best_obj = min(obj_all)
        # store the lowest objective function value
        if best_obj < prev_obj:
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
        obj_iter.append(best_obj)
            # report progress at each iteration
            # print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
    return [best_vector, best_obj, obj_iter, ptype]




