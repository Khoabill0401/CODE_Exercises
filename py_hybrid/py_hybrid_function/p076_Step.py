import math
def p076_Step(x):
    dim = len(x)
    x = x.flatten()
    xsum = 0
    for i in range(dim):
        xsum += math.floor(abs(x[i]))
    return(xsum)
