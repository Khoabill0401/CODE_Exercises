"""
#==================================================================================================#
# Naked Mole Rat Algorithm (NMRA) demo with Appendix A: Test Function Benchmark                    #
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
from py_function.p019_CosineMixture import *
from py_function.p023_Deb import *
from py_function.p030_Exponential import *
from py_function.p032_Griewank import *
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

def NMRA(Function_name, maxiter, n):
    if Function_name == 'p001_Ackley':
        d = 30
        Fun = lambda x: p001_Ackley(x)
        Lb = -35
        Ub = 35
        ptype = 0
    if Function_name == 'p019_CosineMixture':
        d = 2
        Fun = lambda x: p019_CosineMixture( x )
        Lb = -1
        Ub = 1
        ptype = 1
    if Function_name == 'p054_PowellSum':
        d = 30
        Fun = lambda x: p054_PowellSum(x)
        Lb = -1
        Ub = 1
        ptype = 0
    if Function_name == 'p057_Quartic':
        d = 30
        Fun = lambda x: p057_Quartic(x)
        Lb = -1.28
        Ub = 1.28
        ptype = 0
    if Function_name == 'p058_Quintic':
        d = 30
        Fun = lambda x: p058_Quintic(x)
        Lb = -10
        Ub = 10
        ptype = 0
    if Function_name == 'p065_Salomon':
        d = 2
        Fun = lambda x: p065_Salomon(x)
        Lb = -100
        Ub = 100
        ptype = 1
    if Function_name == 'p071_SchumerSteiglitz':
        d = 30
        Fun = lambda x: p071_SchumerSteiglitz(x)
        Lb = 0
        Ub = 10e74
        ptype = 1
    if Function_name == 'p075_Sphere':
        d = 30
        Fun = lambda x: p075_Sphere(x)
        Lb = 0
        Ub = 10
        ptype = 0
    if Function_name == 'p076_Step':
        d = 30
        Fun = lambda x: p076_Step(x)
        Lb = -100
        Ub = 100
        ptype = 1
        '''if Function_name == 'p077_Stepint':
        d = 5
        Fun = lambda x: p077_Stepint(x)
        Lb = -5.12
        Ub = 5.12
        ptype = 1'''
    if Function_name == 'p079_SumSquares':
        d = 2
        Fun = lambda x: p079_SumSquares(x)
        Lb = -10
        Ub = 10
        ptype = 0
    if Function_name == 'p080_StyblinskiTang':
        d = 2
        Fun = lambda x: p080_StyblinskiTang(x)
        Lb = -5
        Ub = 5
        ptype = 1
    if Function_name == 'p096_XinSheYangSecond':
        d = 30
        Fun = lambda x: p096_XinSheYangSecond(x)
        Lb = -2*np.pi
        Ub = 2*np.pi
        ptype = 1
    if Function_name == 'p098_XinSheYangFourth':
        d = 2
        Fun = lambda x: p098_XinSheYangFourth(x)
        Lb = -10
        Ub = 10
        ptype = 1
    if Function_name == 'p100_Zakharov':
        d = 2
        Fun = lambda x: p100_Zakharov(x)
        Lb = -5
        Ub = 10
        ptype = 0
    # n = 50                        # colony size of NMR
    bp = 0.5                        # breeding parameter
    breeders = n/5                  # breeder population
    # workers = n - breeders        # worker population
    iter = 0
    NMRsolution = np.zeros((n, d), dtype = float)
    NMRfitness = np.zeros((n), dtype = float)
    for i in range(n):
        # Lb + (Ub - Lb)*np.random(1, d)
        NMRsolution[i, :] = Lb + (Ub - Lb)*np.random.uniform(0, 1, (1, d))
        NMRfitness[i] = Fun(NMRsolution[i, :])
    # NMRfitness = NMRfitness.sort()
    # Find the current NMRbest
    fmin = NMRfitness.min()
    I = min(range(len(NMRfitness)), key=NMRfitness.__getitem__)
    NMRbest = NMRsolution[I, :]
    S = NMRsolution

    bb = np.zeros((maxiter - 1), dtype = float)
    while (iter < maxiter - 1):         # Loop over worker and breeders
        # For workers
        for i in range(10, n):
            Lambda = random.random()
            # Find random NMR in the neighborhood
            ab = np.random.permutation(n)
            L = Levy(d)
            S[i, :] = (NMRsolution[i, :] + L*(NMRsolution[ab[0], :]) - NMRsolution[ab[1], :])
            Fnew = Fun(S[i, :])
            # If NMRfitness improves (better NMRsolutions found), update then
            if (Fnew <= NMRfitness[i]):
                NMRsolution[i, :] = S[i, :]
                NMRfitness[i] = Fnew
        # For breeders
        for z in range(int(breeders)):
            rand = random.random()
            if rand > bp:
                NMRneighbours = np.random.permutation(10)
                S[z, :] = (1 - Lambda)*S[z, :] + (Lambda*(NMRbest - NMRsolution[NMRneighbours[0], :]))
                # Evaluate new NMRsolutions
                Fnew = Fun(S[z, :])
                # if NMRfitness improves (better NMRsolutions found), update then
                if (Fnew <= NMRfitness[z]):
                    NMRsolution[z, :] = S[z, :]
                    NMRfitness[z] = Fnew
        for i in range(n):
            Flag4Ub = S[i, :] > Ub
            Flag4Lb = S[i, :] < Lb
            S[i, :] = (S[i, :]*np.logical_not(Flag4Ub + Flag4Lb)) + Ub*Flag4Ub + Lb*Flag4Lb
        # NMRfitness = NMRfitness.sort()
        fmin = NMRfitness.min()
        I = min(range(len(NMRfitness)), key=NMRfitness.__getitem__)
        NMRbest = NMRsolution[I, :]
        S = NMRsolution
        bb[iter] = fmin
        iter += 1
    return (NMRbest, fmin, bb, ptype)



