import os, math, operator, sys
import numpy as np

def norm(v):
    '''Calculate the norm of vectors stored in an array'''
    # Example: nx3 array containing the coordinates of n 3D vectors
    # v = np.array([[1, 2, 3]
    #               [...]
    #               [4, 5, 6]])
    # norm(v) will return array([3.74165739, ..., 8.77496439])
    return(np.sqrt(np.sum(v**2,1))[:,None].flatten())

def norml(v):
    '''Normalize vector stored in 1D array'''
    if v.ndim != 1:
        sys.exit('Input array is not 1D!')
    return(v/np.sqrt(np.sum(v**2)))

def Norml(v):
    '''Normalize vectors stored in 2D array'''
    # Example: nx3 array containing the coordinates of n 3D vectors
    # np.array([[1, 2, 3], -------->   array([[0.26726124, 0.53452248, 0.80178373],
    #           [...]
    #           [4, 5, 6]]) ------->          [0.45584231, 0.56980288, 0.68376346]])
    # Works with numpy arrays
    return(v/np.sqrt(np.sum(v**2,1))[:,None])

def SSprod(u,v):
    '''Dot (scalar) product'''
    # u and v are nx3 arrays containing the coordinates of n 3D vectors
    return(np.sum(u*v, 1))

def vprod(u,v):
    '''Cross (vectorial) product taking two 1D arrays (= two vectors)'''
    if u.ndim != 1 and v.ndim != 1:
        sys.exit('Input array(s) is(are) not 1D!')
    return(np.array([u[1]*v[2]-u[2]*v[1],
                     u[2]*v[0]-u[0]*v[2],
                     u[0]*v[1]-u[1]*v[0]]))  # 3d cross (vector) product

def Vprod(u,V):
    '''Cross (vectorial) product taking one vector (u) and an array (V)'''
    return(np.stack((u[1] * V[:, 2]-u[2] * V[:, 1],
              u[2] * V[:, 0]-u[0] * V[:, 2],
              u[0] * V[:, 1]-u[1] * V[:, 0]),axis=1))

def VVprod(U,V):
    '''Cross (vectorial) product for 3D vectors stored in arrays'''
    return(np.stack((U[:,1]*V[:,2]-U[:,2]*V[:,1],
              U[:,2]*V[:,0]-U[:,0]*V[:,2],
              U[:,0]*V[:,1]-U[:,1]*V[:,0]), axis=1))
