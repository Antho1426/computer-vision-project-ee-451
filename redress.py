import numpy as np
from numpy.linalg import eig, inv

"""
    #val_p, v = eig(sigma)
    #ind_max = np.argmax(val_p)
    #ind_min = np.argmin(val_p)
    #mu1 = v[ind_max]
    #mu2 = v[ind_mim]
"""
def principal_axes(im) :
    
    vect_true = np.where(im>0)
    shape = np.array([vect_true[0], vect_true[1]]).T
        
    mu = np.mean(shape, axis=0)
    sigma = np.dot((shape-mu).T, (shape-mu))/shape.shape[1]
    
    l, v = eig(sigma)
    ind_max = np.argmax(l)
    ind_min = np.argmin(l)
    tmp1 = v[:, ind_max]
    tmp2 = v[:, ind_min]
    v[:,0] = tmp1
    v[:,1] = tmp2
    sigma = np.array([[l[ind_max], 0], [0, l[ind_min]]])
    sigma = np.dot(np.dot(v,sigma), inv(sigma))
    
    """
    mu20 = sigma[0][0]
    mu11 = sigma[0][1]
    mu02 = sigma[1][1]
    """
    #alpha = 1/2*np.arctan([2*mu11/(mu20-mu02)])
    alpha = np.arctan(v[1][0]/v[0][0])
    
    return alpha, v
