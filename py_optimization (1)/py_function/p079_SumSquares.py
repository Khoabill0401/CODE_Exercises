
def p079_SumSquares(x):
    dim = len(x)
    x = x.flatten()
    xsum = 0
    for i in range(dim):
        xsum += i*x[i]**2
    o =  xsum
    return(o)

