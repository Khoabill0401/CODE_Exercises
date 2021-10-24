import numpy as np
def p057_Quartic(x):
    dim = len(x)
    x = x.flatten()
    xsum4 = 0
    for i in range(dim):
        xsum4 += i*(x[i]**4)
    o = xsum4 + np.random.uniform(0,1)
    return (o)