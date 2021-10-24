def p080_StyblinskiTang(x):
    dim = len(x)
    x = x.flatten()
    xsum4 = 0
    xsum2 = 0
    xsum  = 0
    for i in range(dim):
        xsum4 += x[i]**4
        xsum2 += x[i]**2
        xsum  += x[i]
    o = 0.5*(xsum4 - 16*xsum2 + 5*xsum)
    return (o)

