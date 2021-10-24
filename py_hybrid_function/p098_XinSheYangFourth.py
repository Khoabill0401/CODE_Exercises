import numpy as np
def p098_XinSheYangFourth(x):
    dim = len(x)
    x = x.flatten()
    xsum2 = 0
    xsumsin = 0
    xsumsin2 = 0
    for i in range(dim):
        xsum2 += x[i]**2
        xsumsin += np.sin(np.pi*x[i])**2
        xsumsin2 += np.sin(np.pi*np.sqrt(abs(x[i])))**2
    o =  (xsumsin-np.exp(-xsum2))*np.exp(-xsumsin2)
    return(o)

