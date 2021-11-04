

import numpy as np  # Normal NumPy import
cimport numpy as cnp  # Import for NumPY C-API
import cython
from libc.math cimport floor, tan, M_PI, pow, sqrt
@cython.boundscheck(False)
@cython.wraparound(False)


# Coordinates of symmetrical point
cpdef cnp.ndarray[cnp.uint16_t, ndim=1] define_sym_point(cnp.ndarray[cnp.uint16_t, ndim=1] pt,
            cnp.ndarray[cnp.float64_t, ndim=1] vec,
            cnp.ndarray[cnp.uint16_t, ndim=1] refpt):
    # Compute projection of pt on axis defined by refpt and vec
    cdef cnp.ndarray[cnp.float64_t, ndim=1] proj_pt = np.empty(2, dtype=np.float64)
    cdef double factor = (pt[0] - refpt[0]) * vec[0] + (pt[1] - refpt[1]) * vec[1]
    proj_pt[0] = refpt[0] + vec[0] * factor
    proj_pt[1] = refpt[1] + vec[1] * factor
    # Compute symmetrical point
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] sym_pt = np.empty(2, dtype=np.uint16)
    sym_pt[0] = int(floor(2 * proj_pt[0] - pt[0]))
    sym_pt[1] = int(floor(2 * proj_pt[1] - pt[1]))
    return(sym_pt)

cpdef define_symvec(int angle):
    # angle in degrees
    cdef cnp.ndarray[cnp.float64_t, ndim=1] tmp = np.empty(2, dtype=np.float64)
    tmp[0] = - tan(angle * M_PI / 180)
    tmp[1] = 1
    cdef double s = sqrt(pow(tmp[0],2) + pow(tmp[1],2))
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vec = np.empty(2, dtype=np.float64)
    vec[0] = tmp[0]/s
    vec[1] = tmp[1]/s
    return(vec)

# Compute mirror image along axis defined by refpt and vec
cpdef compute_mirror_image(cnp.ndarray[cnp.uint8_t, ndim=2] im,
            cnp.ndarray[cnp.uint16_t, ndim=1] refpt,
            cnp.ndarray[cnp.float64_t, ndim=1] vec):
    # im: grayscale image
    # vec: vector defining symmetry axis (in opencv coordinates)
    # refpt: reference point (in opencv coordinates)
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] im_out = np.zeros_like(im, dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] pt = np.empty(2, dtype=np.uint16)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] proj_pt = np.empty(2, dtype=np.float64)
    cdef double factor = 0
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] sym_pt = np.empty(2, dtype=np.uint16)
    cdef int height = im.shape[0]
    cdef int width = im.shape[1]
    cdef int counter = 0  # Counts pixels which are both at 255
    cdef int i, j
    #
    for j in range(height):
        for i in range(width):
            pt[0] = i; pt[1] = j
            # Compute projection of pt on axis defined by refpt and vec
            factor = (pt[0] - refpt[0]) * vec[0] + (pt[1] - refpt[1]) * vec[1]
            proj_pt[0] = refpt[0] + vec[0] * factor
            proj_pt[1] = refpt[1] + vec[1] * factor
            # Compute symmetrical point
            sym_pt[0] = int(floor(2 * proj_pt[0] - pt[0]))
            sym_pt[1] = int(floor(2 * proj_pt[1] - pt[1]))
            if ((sym_pt[0] >= 0) & (sym_pt[0] < width) & (sym_pt[0] >= 1) & (sym_pt[1] < height)):
                im_out[j,i] = im[sym_pt[1],sym_pt[0]]
                if ((im[j,i] == 255) & (im_out[j,i] == 255)):
                    counter += 1
    return im_out, counter

# Compute best symmetry axis
cpdef compute_sym_axis(cnp.ndarray[cnp.uint8_t, ndim=2] im,
            cnp.ndarray[cnp.uint16_t, ndim=1] refpt,
            cnp.ndarray[cnp.int16_t, ndim=1] angle_range,
            cnp.ndarray[cnp.int16_t, ndim=1] hoffset_range):
    # im: 8 bit grayscale image
    # vec: vector defining symmetry axis (in opencv coordinates)
    # refpt: reference point (in opencv coordinates)
    #
    # Definitions
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] im_out = np.zeros_like(im, dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] pt = np.empty(2, dtype=np.uint16)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] proj_pt = np.empty(2, dtype=np.float64)
    cdef double factor = 0
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] sym_pt = np.empty(2, dtype=np.uint16)
    cdef int height = im.shape[0]
    cdef int width = im.shape[1]
    cdef int counter = 0  # Counts pixels which are both at 255
    cdef cnp.ndarray[cnp.uint16_t, ndim=2] counts = np.zeros((angle_range[1] - angle_range[0], hoffset_range[1] - hoffset_range[0]), dtype=np.uint16)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] tmp = np.empty(2, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vec = np.empty(2, dtype=np.float64)
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] new_refpt = np.zeros(2, dtype=np.uint16)
    cdef double s
    cdef int i, j, ang, m, offset
    #
    for ang in range(angle_range[1] - angle_range[0]):
        # Compute vector defining symmetry axis (vec)
        tmp[0] = - tan((ang + angle_range[0]) * M_PI / 180)
        tmp[1] = 1
        s = sqrt(pow(tmp[0],2) + pow(tmp[1],2))
        vec[0] = tmp[0]/s
        vec[1] = tmp[1]/s
        for offset in range(hoffset_range[1] - hoffset_range[0]):
            new_refpt[0] = refpt[0] + offset + hoffset_range[0]
            new_refpt[1] = refpt[1]
            counter = 0
            for j in range(height):
                for i in range(width):
                    pt[0] = i; pt[1] = j
                    # Compute projection of pt on axis defined by refpt and vec
                    factor = (pt[0] - new_refpt[0]) * vec[0] + (pt[1] - new_refpt[1]) * vec[1]
                    proj_pt[0] = new_refpt[0] + vec[0] * factor
                    proj_pt[1] = new_refpt[1] + vec[1] * factor
                    # Compute symmetrical point
                    sym_pt[0] = int(floor(2 * proj_pt[0] - pt[0]))
                    sym_pt[1] = int(floor(2 * proj_pt[1] - pt[1]))
                    if ((sym_pt[0] >= 0) & (sym_pt[0] < width) & (sym_pt[0] >= 1) & (sym_pt[1] < height)):
                        im_out[j,i] = im[sym_pt[1],sym_pt[0]]
                        if ((im[j,i] == 255) & (im_out[j,i] == 255)):
                            counter += 1
            counts[ang,offset] = counter
    return counts
