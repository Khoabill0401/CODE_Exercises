"""
# ==================================================================================== #
# This work has been done by:                                                          #
# Nguyen Anh Khoa  - 1810240                                                           #
# Phone: 0868.840.441                                                                  #
# Email: khoa.nguyen41@hcmut.edu.vn                                                    #
# ==================================================================================== #
"""
import numpy as np

def simplebounds(s, Lb, Ub):

    Flag4Ub = s > Ub
    Flag4Lb = s < Lb
    s = s*np.logical_not(Flag4Ub + Flag4Lb) + Ub*Flag4Ub + Lb*Flag4Lb

    return (s)