"""
%====================================================================================%
% Runs the DE_TCR optimization algorithm                                             %
%====================================================================================%
"""
import math
import numpy as np
from numpy import ones
from numpy import zeros
from numpy.random import rand
from scipy.stats import iqr

def DE_TCR():
    #------------------------------------------------------------------#
    # Problem to solve

    NVAR = 30                      # Number of decision variables
    Generations = 2e2*NVAR         # Generations
    # (Re) Initialization bounds
    Bounds = [-5.12*ones((NVAR, 1), dtype = float), 5.12*ones((NVAR, 1), dtype = float)]
    # Optimization Bounds
    Initial = [-5.12*ones((NVAR, 1), dtype = float), 5.12*ones((NVAR, 1), dtype = float)]
    # Cost function
    sop = 'CostFunction'

    #------------------------------------------------------------------#
    # Local Search

    # Probability to run a Local Search procedure in a given Child
    RateLS = 1 - 1/(100*NVAR)

    #------------------------------------------------------------------#
    # Adaptive Mechanism

    MaxCr = [0.2, 1]               # Maximum Crossover rate
    MedCr = [0.2, 0.5]             # Median Crossover rate
    MinCr = [0.2, 0.2]             # Minimum Crossver rate
    MaxF = [0.3, 0.5]              # Maximum Scaling Factor
    MedF = [0.3, 0.4]              # Median Scaling Factor
    MinF = [0.3, 0.3]              # Minimum Scaling Factor

    # Linear recombination factor
    recombination = 0.75
    # Threshold for CR successes
    CRsuccess = 15

    #------------------------------------------------------------------#
    # Population Management

    Xpop = 5*NVAR                  # Population size
    GammaVar = 3                   # Interquartil difference for refresh
    # Minimum diversity in population
    minVarPop = 0.05*(Initial[:, 1] - Initial[:, 0]).T
    XpopRefresh = 2*NVAR           # Population Refresh size

    #------------------------------------------------------------------#
    # Initialization of variables

    # Adaptive Mechanism

    # To record successful values on CR
    ParamEVO = zeros((CRsuccess, 1), dtype = float)
    # To record successes on CR
    SuccessEvoCr = 0

    # Population

    Child = zeros((Xpop, NVAR), dtype= float)       # Child population
    Parent = zeros((Xpop, NVAR), dtype= float)      # Parent population
    Mutant = zeros((Xpop, NVAR), dtype= float)      # Mutant population
    JxParent = zeros((Xpop, 1), dtype= float)       # Fitness of Parents Population

    #------------------------------------------------------------------#
    # Population for evoluntionary process

    PopParent = zeros((Xpop, NVAR), dtype= float)   # Population initialization

    for xpop in Xpop:                               # Uniform distribution
        for nvar in NVAR:
            PopParent[xpop, nvar] = Initial[nvar, 0] + rand()*(Initial[nvar, 1] - Initial[nvar, 0])

    JxPopParent = sop
    FES = Xpop

    #------------------------------------------------------------------#
    # Evolution process

    for n in Generations:
        CounterGen = n
        DE_TCR.JxPopParent = JxPopParent
        DE_TCR.JxParent = JxParent
        TCR = (MinCr, MedCr, MaxCr)

    # ------------------------------------------------------------------#
    # Population Refreshment Mechanism

        VarPop = iqr(PopParent) # Measuring the interquartile range difference

        if VarPop <= ((Initial[:, 1] - Initial[:, 0]).T)/GammaVar: # We need a refresh
   #         LookingBest =






