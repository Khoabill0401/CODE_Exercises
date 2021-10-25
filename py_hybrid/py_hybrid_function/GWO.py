"""
# ==================================================================================== #
# Grey Wolf Optimizer                                                                  #
# ==================================================================================== #
# This work has been done by:                                                          #
# Nguyen Anh Khoa  - 1810240                                                           #
# Phone: 0868.840.441                                                                  #
# Email: khoa.nguyen41@hcmut.edu.vn                                                    #
# ==================================================================================== #
# Provided Benchmark functions that we have tested:                                    #
# 001. p001_Ackley: Ackley Function                                                    #
# 002. p002_Alpine: Alpine Function                                                    #
# 003. p003_BartelsConn: Bartels Conn Function                                         #
# 005. p005_Bird: Bird Function                                                        #
# 006. p006_Bohachevsky: Bohachevsky Function                                          #
# 007. p007_Booth: Booth Function                                                      #
# 010. p010_Brent: Brent Function                                                      #
# 011. p011_Brown: Brown Function                                                      #
# 012. p012_Bukin: Bukin Function                                                      #
# 019. p019_CosineMixture: Cosine Mixture Function                                     #
# 020. p020_Csendes: Csendes Function                                                  #
# 023. p023_Deb: Deb Function                                                          #
# 030. p030_Exponential: Exponential Function                                          #
# 032. p032_Griewank: Griewank Function                                                #
# 037. p037_Hosaki: Hosaki Function                                                    #
# 045. p045_Mishra: Mishra Function                                                    #
# 049. p049_Paviani: Paviani Function                                                  #
# 054. p054_PowellSum: PowellSum Function                                              #
# 057. p057_Quartic: Quartic Function                                                  #
# 058. p058_Quintic: Quintic Function                                                  #
# 065. p065_Salomon: Salomon Function                                                  #
# 071. p071_SchumerSteiglitz: SchumerSteiglitz Function                                #
# 075. p075_Sphere: Sphere Function                                                    #
# 076. p076_Step: Step Function                                                        #
# 077. p077_Stepint: Stepint Function                                                  #
# 079. p079_SumSquares: SumSquares Function                                            #
# 080. p080_StyblinskiTang: Styblinski - Tang Function                                 #
# 096. p096_XinSheYang2: XinSheYangSecond Function                                     #
# 098. p098_XinSheYang4: XinSheYangFourth Function                                     #
# 100. p100_Zakharov: Zakharov Function                                                #
# ==================================================================================== #
# ** Test for 10 - bar problem                                                         #
# 101. solve10bar                                                                      #
# ==================================================================================== #
"""
import math
from math import inf
import numpy as np
from numpy import zeros
from numpy import logical_not
from numpy.random import rand
import random
from py_hybrid_function.initialization import *
from py_hybrid_function.p001_Ackley import *
from py_hybrid_function.p002_Alpine import *
from py_hybrid_function.p003_BartelsConn import *
from py_hybrid_function.p005_Bird import *
from py_hybrid_function.p006_Bohachevsky import *
from py_hybrid_function.p007_Booth import *
from py_hybrid_function.p010_Brent import *
from py_hybrid_function.p011_Brown import *
from py_hybrid_function.p012_Bukin import *
from py_hybrid_function.p019_CosineMixture import *
from py_hybrid_function.p020_Csendes import *
from py_hybrid_function.p023_Deb import *
from py_hybrid_function.p030_Exponential import *
from py_hybrid_function.p032_Griewank import *
from py_hybrid_function.p037_Hosaki import *
from py_hybrid_function.p045_Mishra import *
from py_hybrid_function.p049_Paviani import *
from py_hybrid_function.p054_PowellSum import *
from py_hybrid_function.p057_Quartic import *
from py_hybrid_function.p058_Quintic import *
from py_hybrid_function.p065_Salomon import *
from py_hybrid_function.p071_Schumer import *
from py_hybrid_function.p075_Sphere import *
from py_hybrid_function.p076_Step import *
from py_hybrid_function.p077_Stepint import *
from py_hybrid_function.p079_SumSquares import *
from py_hybrid_function.p080_StyblinskiTang import *
from py_hybrid_function.p096_XinSheYang2 import *
from py_hybrid_function.p098_XinSheYang4 import *
from py_hybrid_function.p100_Zakharov import *
from py_hybrid_function.solve10bar import *

def GWO(Function_name, Max_iteration, SearchAgents_no):

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
        Lb = -2 * math.pi
        Ub = 2 * math.pi
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
    if Function_name == 'solve10bar':
        d = 10 # number of design variables
        Fun = lambda x: solve10bar(x)
        Lb = 0.645e-4*np.ones((1, d), dtype=float).flatten()
        Ub = 50e-4*np.ones((1, d), dtype=float).flatten()
        ptype = 1
        tol = 1e-6  # Tolerance for the stopping criterion

    # Main calculation
    dim = d
    # Initialize alpha, beta and delta_pos
    Alpha_pos = zeros((1, dim), dtype = float)
    Beta_pos  = zeros((1, dim), dtype = float)
    Delta_pos = zeros((1, dim), dtype = float)
    Alpha_score = float(inf)    # change this to - float(math.inf) for maximization problems
    Beta_score  = float(inf)
    Delta_score = float(inf)

    # Initialize the positions of search agents
    #if Function_name == "solve10bar":
    #    # Target vector
    #    x = np.matlib.repmat(Lb, SearchAgents_no, 1) + np.random.uniform(0, 1, (SearchAgents_no, dim)) * np.matlib.repmat(Ub - Lb, SearchAgents_no, 1)

    #    # Evaluate the objective function w.r.t constraints
    #    (Positions) = Fun(x)
    #else: Positions = initialization(SearchAgents_no, dim, Ub, Lb)

    Positions = initialization(SearchAgents_no, dim, Ub, Lb)

    Convergence_curve = zeros((Max_iteration), dtype = float)
    l = 0  # Loop counter

    # Main loop
    for i in range(Max_iteration):
        for j in range(len(Positions)):

            # Return back the search agents that go beyond the boundaries of the search space
            Flag4Ub = Positions[j, :] > Ub
            Flag4Lb = Positions[j, :] < Lb
            Positions[j, :] = (Positions[j, :]*logical_not(Flag4Ub + Flag4Lb)) + (Ub*Flag4Ub) + (Lb*Flag4Lb)

            # Calculate objective function for each search agent
            fitness = Fun(Positions[j, :]).copy()

            # Update Alpha, Beta and Delta
            if Function_name == 'solve10bar':
                for iter in range(SearchAgents_no):
                    if fitness[iter] < Alpha_score:
                        # Update Alpha
                        Alpha_score = fitness[iter].copy()
                        Alpha_pos = Positions[j, :].copy()

                    if (fitness[iter] > Alpha_score) and (fitness[iter] < Beta_score):
                        # Update Beta
                        Beta_score = fitness[iter].copy()
                        Beta_pos = Positions[j, :].copy()

                    if (fitness[iter] > Alpha_score) and (fitness[iter] > Beta_score) and (fitness[iter] < Delta_score):
                        # Update Delta
                        Delta_score = fitness[iter].copy()
                        Delta_pos = Positions[j, :].copy()
            else:
                if fitness < Alpha_score:
                    # Update Alpha
                    Alpha_score = fitness.copy()
                    Alpha_pos = Positions[j, :].copy()

                if (fitness > Alpha_score) and (fitness < Beta_score):
                    # Update Beta
                    Beta_score = fitness.copy()
                    Beta_pos = Positions[j, :].copy()

                if (fitness > Alpha_score) and (fitness > Beta_score) and (fitness < Delta_score):
                    # Update Delta
                    Delta_score = fitness.copy()
                    Delta_pos = Positions[j, :].copy()


        # a decreases linearly from 2 to 0
        a = 2 - l*(2/Max_iteration)

        # Update the Position of search agents including omegas
        for ii in range(len(Positions)):
            for j in range(len(Positions[0])):

                r1 = rand()  # r1 is a random number in [0, 1]
                r2 = rand()  # r1 is a random number in [0, 1]

                A1 = 2*a*r1 - a # Equation (3.3)
                C1 = 2*r2       # Equation (3.4)

                # Equation (3.5) - Part 1
                D_alpha = abs(C1*Alpha_pos[j] - Positions[ii, j])
                # Equation (3.6) - Part 1
                X1 = Alpha_pos[j] - A1*D_alpha

                r1 = rand()  # r1 is a random number in [0, 1]
                r2 = rand()  # r1 is a random number in [0, 1]

                A2 = 2 * a * r1 - a  # Equation (3.3)
                C2 = 2 * r2  # Equation (3.4)

                # Equation (3.5) - Part 2
                D_beta = abs(C2 * Beta_pos[j] - Positions[ii, j])
                # Equation (3.6) - Part 2
                X2 = Beta_pos[j] - A2 * D_beta

                r1 = rand()  # r1 is a random number in [0, 1]
                r2 = rand()  # r1 is a random number in [0, 1]

                A3 = 2 * a * r1 - a  # Equation (3.3)
                C3 = 2 * r2  # Equation (3.4)

                # Equation (3.5) - Part 3
                D_delta = abs(C3 * Delta_pos[j] - Positions[ii, j])
                # Equation (3.6) - Part 3
                X3 = Delta_pos[j] - A3 * D_delta

                # Equation (3.7)
                Positions[ii, j] = (X1 + X2 + X3)/3

        Convergence_curve[l] = Alpha_score.copy()
        l += 1

    return (Alpha_score, Alpha_pos, Convergence_curve, ptype)