import numpy as np
from numpy.linalg import eig, inv


def region_growing(image,seed,threshold) :
    """
    Function : Implementation of a region growing algorithm from a known starting pixel (seed)
    
    Inputs : - image : Picture to analyze (need to be unicolor)
             - seed : Starting pixel (need to be an 1x2 array)
             - threshold : Accepted deviation between the intensity of the neighbors (pixels)
    Output : - region : List containing all the pixels into the region composed by the pixels 
                with approximatively the same intensity                
    """
    
    #neighbours = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])     # List of the neighbors to check (Manhattan distance)
    neighbours = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (-1,-1), (1,1), (-1,1), (1,-1)])
    
    P_checked = np.zeros(image.shape)     # Matrix used to look if the pixel has already been checked
    P_checked[tuple(seed)] = 1
    
    check_list = np.array([seed])     # List of the pixel to check later
    region = np.array([seed])     # List of the pixels included into the region
    
    while np.size(check_list,axis=0) != 0 :
        
        host = check_list[-1]     # Coordinates of the host pixel
        host_intensity = image[tuple(host)]     # Intensity of the host pixel 
        
        for n in neighbours :     # Loop iterating over the neighbours of the host pixel
            neigh = host + n     # Coordinates of the neighbor
            if neigh[0] < image.shape[0] and neigh[1] < image.shape[1] :     # Checking if the analyzed pixel is in the image
                
                neigh_intensity = image[tuple(neigh)]     # Intensity of a neighbor of the host pixel
            
                if abs(float(neigh_intensity) - float(host_intensity)) < threshold and P_checked[tuple(neigh)] == 0 :     # P_checked[i][j] = 0 means that the pixel in coordinates [i][j] has not been checked yet
                    region = np.insert(region,-1,neigh,axis=0)     # Insertion of the neighbor in the region
                
                    check_list = np.insert(check_list,0,neigh,axis=0)     # Insertion of the neighbor in the pixel to check after (beginning of the list)
                    P_checked[tuple(neigh)] = 1
                
        check_list = np.delete(check_list,(-1),axis=0)
    
    return region



def principal_axes(im) :
    """
    Function that find the angle of the axes of inertia
    
    Input : - im --> Binary image to process
    
    Outputs : - alpha --> angle for the rotation
              - v --> eigenvector
    """
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

    alpha = np.arctan(v[1][0]/v[0][0])
    
    return alpha, v