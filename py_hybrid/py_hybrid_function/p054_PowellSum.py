def p054_PowellSum(x):
    dim = len(x)
    x = x.flatten()
    xsum = 0
    for i in range(dim):
        xsum += abs(x[i])**(i+1)
    o = xsum
    return (o)
