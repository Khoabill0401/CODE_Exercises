def p075_Sphere(x):
    dim = len(x)
    x = x.flatten()
    xsum = 0
    for i in range(dim):
        xsum += x[i]**2
    return (xsum)

