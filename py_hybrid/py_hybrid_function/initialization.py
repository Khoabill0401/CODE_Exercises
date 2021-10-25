"""
# ==================================================================================== #
# This work has been done by:                                                          #
# Nguyen Anh Khoa  - 1810240                                                           #
# Phone: 0868.840.441                                                                  #
# Email: khoa.nguyen41@hcmut.edu.vn                                                    #
# ==================================================================================== #
"""
import numpy as np

def initialization(SearchAgents_no, dim, Ub, Lb):

    if isinstance(Ub, (int, float)):
        Boundary_no = 1
    else:
        Boundary_no = len(Ub)
    # If the boundaries of all variables are equal and user enter a single
    # number for both Ub and Lb
    Positions = np.zeros((SearchAgents_no, dim), dtype=float)

    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim)*(Ub - Lb) + Lb

    # If each variable has a different Ub and Lb
    if Boundary_no > 1:
        for i in range(dim):
            Ub_i = Ub[i].copy()
            Lb_i = Lb[i].copy()
            Positions[:, i] = (np.random.rand(SearchAgents_no, 1)*(Ub_i - Lb_i) + Lb_i).ravel()

    return (Positions)