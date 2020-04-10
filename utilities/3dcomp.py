

class 3dc:
    def __init__(self):
        def norm(v):
            '''Calculate the norm of vectors stored in an array'''
            # Example: nx3 array containing the coordinates of n 3D vectors
            # v = np.array([[1, 2, 3]
            #               [...]
            #               [4, 5, 6]])
            # norm(v) will return array([3.74165739, ..., 8.77496439])
            return (np.sqrt(np.sum(v ** 2, 1))[:, None].flatten())
        def Vproj(u,V):
            """Compute projection of n 3D vectors stored in V (nx3 array) onto plane normal to vector u"""
            # u: vector defining projection plane (reference vector), e.g. np.array([1,2,3])
            # V: 3-column 2D array containing vector coordinates
            un = utils.norml(u)  # normalize reference vector
            v = utils.Norml(-utils.Vprod(un,utils.Norml(utils.Vprod(un,utils.Norml(V)))))
            s = utils.SSprod(v, V)
            return(s[:, None] * v)
