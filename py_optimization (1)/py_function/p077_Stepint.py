import math
def p077_Stepint(x):
    dim = len(x)
    x = x.flatten()
    xsum = 0
    for i in range(dim):
        xsum += math.floor(x[i])
    o = 25 + xsum
    return(o)