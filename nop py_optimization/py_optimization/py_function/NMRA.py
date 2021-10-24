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
import copy
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
from py_function.p049_Paviani import *
from py_function.p054_PowellSum import *
from py_function.p057_Quartic import *
from py_function.p058_Quintic import *
from py_function.p065_Salomon import *
from py_function.p071_Schumer import *
from py_function.p075_Sphere import *
from py_function.p076_Step import *
from py_function.p077_Stepint import *
from py_function.p079_SumSquares import *
from py_function.p080_StyblinskiTang import *
from py_function.p096_XinSheYang2 import *
from py_function.p098_XinSheYang4 import *
from py_function.p100_Zakharov import *

def NMRA(Function_name, maxiter, n):
    # Determine what benchmark function
    if Function_name == 'p001_Ackley':
        d = 30
        Fun = lambda x: p001_Ackley(x)
        Lb = -35
        Ub = 35
        ptype = 1
    if Function_name == 'p002_Alpine':
        d = 20
        Fun = lambda x: p002_Alpine(x)
        Lb = -10
        Ub = 10
        ptype = 1
    if Function_name == 'p003_BartelsConn':
        d = 2
        Fun = lambda x: p003_BartelsConn(x)
        Lb = -500
        Ub = 500
        ptype = 1
    if Function_name == 'p005_Bird':
        d = 2
        Fun = lambda x: p005_Bird(x)
        Lb = -2 * np.pi
        Ub = 2 * np.pi
        ptype = 1
    if Function_name == 'p006_Bohachevsky':
        d = 2
        Fun = lambda x: p006_Bohachevsky(x)
        Lb = -100
        Ub = 100
        ptype = 1
    if Function_name == 'p007_Booth':
        d = 2
        Fun = lambda x: p007_Booth(x)
        Lb = -10
        Ub = 10
        ptype = 1
    if Function_name == 'p010_Brent':
        d = 2
        Fun = lambda x: p010_Brent(x)
        Lb = -10
        Ub = 10
        ptype = 1
    if Function_name == 'p011_Brown':
        d = 15
        Fun = lambda x: p011_Brown(x)
        Lb = -1
        Ub = 4
        ptype = 1
    if Function_name == 'p012_Bukin':
        d = 2
        Fun = lambda x: p012_Bukin(x)
        Lb = np.array([-15, -3])
        Ub = np.array([-5, 3])
        ptype = 1
    if Function_name == 'p019_CosineMixture':
        d = 2
        Fun = lambda x: p019_CosineMixture(x)
        Lb = -1
        Ub = 1
        ptype = 1
    if Function_name == 'p020_Csendes':
        d = 30
        Fun = lambda x: p020_Csendes(x)
        Lb = -1
        Ub = 1
        ptype = 1
    if Function_name == 'p023_Deb':
        d = 10
        Fun = lambda x: p023_Deb(x)
        Lb = -1
        Ub = 1
        ptype = 1
    if Function_name == 'p030_Exponential':
        d = 20
        Fun = lambda x: p030_Exponential(x)
        Lb = -1
        Ub = 1
        ptype = 1
    if Function_name == 'p032_Griewank':
        d = 10
        Fun = lambda x: p032_Griewank(x)
        Lb = -100
        Ub = 100
        ptype = 1
    if Function_name == 'p037_Hosaki':
        d = 2
        Fun = lambda x: p037_Hosaki(x)
        Lb = np.array([0, 0])
        Ub = np.array([5, 6])
        ptype = 1
    if Function_name == 'p045_Mishra':
        d = 5
        Fun = lambda x: p045_Mishra(x)
        Lb = -10
        Ub = 10
        ptype = 1
    if Function_name == 'p049_Paviani':
        d = 5
        Fun = lambda x: p049_Paviani(x)
        Lb = 3
        Ub = 9
        ptype = 0
    if Function_name == 'p054_PowellSum':
        d = 20
        Fun = lambda x: p054_PowellSum(x)
        Lb = -1
        Ub = 1
        ptype = 1
    if Function_name == 'p057_Quartic':
        d = 30
        Fun = lambda x: p057_Quartic(x)
        Lb = -1.28
        Ub = 1.28
        ptype = 1
    if Function_name == 'p058_Quintic':
        d = 10
        Fun = lambda x: p058_Quintic(x)
        Lb = -10
        Ub = 10
        ptype = 1
    if Function_name == 'p065_Salomon':
        d = 20
        Fun = lambda x: p065_Salomon(x)
        Lb = -100
        Ub = 100
        ptype = 1
    if Function_name == 'p071_Schumer':
        d = 20
        Fun = lambda x: p071_Schumer(x)
        Lb = -10
        Ub = 10
        ptype = 0
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
    if Function_name == 'p077_Stepint':
        d = 30
        Fun = lambda x: p077_Stepint(x)
        Lb = -5.12
        Ub = 5.12
        ptype = 1
    if Function_name == 'p079_SumSquares':
        d = 30
        Fun = lambda x: p079_SumSquares(x)
        Lb = -10
        Ub = 10
        ptype = 1
    if Function_name == 'p080_StyblinskiTang':
        d = 2
        Fun = lambda x: p080_StyblinskiTang(x)
        Lb = -5
        Ub = 5
        ptype = 1
    if Function_name == 'p096_XinSheYang2':
        d = 2
        Fun = lambda x: p096_XinSheYang2(x)
        Lb = -2*math.pi
        Ub = 2*math.pi
        ptype = 1
    if Function_name == 'p098_XinSheYang4':
        d = 10
        Fun = lambda x: p098_XinSheYang4(x)
        Lb = -10
        Ub = 10
        ptype = 1
    if Function_name == 'p100_Zakharov':
        d = 2
        Fun = lambda x: p100_Zakharov(x)
        Lb = -5
        Ub = 5
        ptype = 1

    # Main calculation
    # n = 50                        # colony size of NMR
    bp = 0.5                        # breeding parameter
    breeders = n/5                  # breeder population
    # workers = n - breeders        # worker population
    #iter = 0
    NMRsolution = np.zeros((n, d), dtype = float)
    NMRfitness = np.zeros((n), dtype = float)
    for i in range(n):
        # Lb + (Ub - Lb)*np.random(1, d)
        NMRsolution[i, :] = Lb + (Ub - Lb)*(np.random.uniform(0, 1, (d, 1))).T
        NMRfitness[i] = Fun(NMRsolution[i, :])
    # NMRfitness = NMRfitness.sort()
    # Find the current NMRbest

    #NMRfitness = np.array([20.3282472030341,	14.3136152015144,	21.6516635811573,	20.5754758966815,	19.7909915921096])
    #NMRsolution = (np.array([[11.4720057688999, 5.90190638448075, -25.7419070071079, 26.1242151488014, -13.1840761015940]])).T

    fmin = NMRfitness.min()
    I = np.argmin(NMRfitness)
    #I = min(range(len(NMRfitness)), key=NMRfitness.__getitem__)
    NMRbest = NMRsolution[I, :].copy()
    S = NMRsolution.copy()

    bb = np.zeros((maxiter - 1), dtype = float)
    for iter in range(maxiter-1):
    #while (iter < maxiter - 1):         # Loop over worker and breeders
        # For workers
        for i in range(int(breeders)-1, n):
            # Find random NMR in the neighborhood
            ab = (np.random.permutation(n))

            #ab = np.array([4, 3, 2, 1, 5]) -1

            L = Levy(d)
            S[i, :] = (NMRsolution[i, :] + L*(NMRsolution[ab[0], :] - NMRsolution[ab[1], :]))
            Fnew = copy.copy(Fun(S[i, :]))
            # If NMRfitness improves (better NMRsolutions found), update then
            if (Fnew <= NMRfitness[i]):
                NMRsolution[i, :] = copy.copy(S[i, :])
                NMRfitness[i] = copy.copy(Fnew)
        # For breeders
        Lambda = random.random()

        #Lambda = 0.376152200922196

        for z in range(int(breeders)):
            rand = random.random()

            #rand = 0.554525145898807

            if rand > bp:
                NMRneighbours = np.random.permutation(int(breeders))

                #NMRneighbours = np.array([1])-1

                S[z, :] = (1 - Lambda)*(S[z, :]) + (Lambda*(NMRbest - NMRsolution[NMRneighbours[0], :]))
                # Evaluate new NMRsolutions
                Fnew = copy.copy(Fun(S[z, :]))
                # if NMRfitness improves (better NMRsolutions found), update then
                if (Fnew <= NMRfitness[z]):
                    NMRsolution[z, :] = copy.copy(S[z, :])
                    NMRfitness[z] = copy.copy(Fnew)


        for i in range(n):
            Flag4Ub = (S[i, :] > Ub)
            Flag4Lb = (S[i, :] < Lb)
            S[i, :] = (S[i, :]*(np.logical_not(Flag4Ub + Flag4Lb))) + (Ub*Flag4Ub) + (Lb*Flag4Lb)
        #NMRfitness = NMRfitness.sort()
        fmin = NMRfitness.min()
        I = min(range(len(NMRfitness)), key=NMRfitness.__getitem__)
        NMRbest = copy.copy(NMRsolution[I, :])
        S = copy.copy(NMRsolution)
        bb[iter] = fmin
        #iter += 1

    return (NMRbest, fmin, bb, ptype)



