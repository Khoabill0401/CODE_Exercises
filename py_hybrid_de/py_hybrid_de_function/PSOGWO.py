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
from py_hybrid_de_function.initialization import *
from py_hybrid_de_function.p001_Ackley import *
from py_hybrid_de_function.p002_Alpine import *
from py_hybrid_de_function.p003_BartelsConn import *
from py_hybrid_de_function.p005_Bird import *
from py_hybrid_de_function.p006_Bohachevsky import *
from py_hybrid_de_function.p007_Booth import *
from py_hybrid_de_function.p010_Brent import *
from py_hybrid_de_function.p011_Brown import *
from py_hybrid_de_function.p012_Bukin import *
from py_hybrid_de_function.p019_CosineMixture import *
from py_hybrid_de_function.p020_Csendes import *
from py_hybrid_de_function.p023_Deb import *
from py_hybrid_de_function.p030_Exponential import *
from py_hybrid_de_function.p032_Griewank import *
from py_hybrid_de_function.p037_Hosaki import *
from py_hybrid_de_function.p045_Mishra import *
from py_hybrid_de_function.p049_Paviani import *
from py_hybrid_de_function.p054_PowellSum import *
from py_hybrid_de_function.p057_Quartic import *
from py_hybrid_de_function.p058_Quintic import *
from py_hybrid_de_function.p065_Salomon import *
from py_hybrid_de_function.p071_Schumer import *
from py_hybrid_de_function.p075_Sphere import *
from py_hybrid_de_function.p076_Step import *
from py_hybrid_de_function.p077_Stepint import *
from py_hybrid_de_function.p079_SumSquares import *
from py_hybrid_de_function.p080_StyblinskiTang import *
from py_hybrid_de_function.p096_XinSheYang2 import *
from py_hybrid_de_function.p098_XinSheYang4 import *
from py_hybrid_de_function.p100_Zakharov import *
from py_hybrid_de_function.solve10bar import *

def PSOGWO(Function_name, Max_iteration, SearchAgents_no, count1):

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
