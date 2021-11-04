import os, pydicom
import numpy as np
from operator import itemgetter

def load_stack(srcdir):
    """read dicom images from a given folder and put them in a 3D array"""
    print('Reading dicom files and calculating 3d array... ', end='')
    files = [os.path.join(srcdir, fname) for fname in os.listdir(srcdir) if fname.endswith('.dcm')]
    # Read slices as a list before sorting
    dcm_slices = [pydicom.read_file(fname) for fname in files]
    # Extract position for each slice to sort and calculate slice spacing
    dcm_slices = [(dcm, thru_plane_position(dcm)) for dcm in dcm_slices]
    dcm_slices = sorted(dcm_slices, key=itemgetter(1))
    spacings = np.diff([dcm_slice[1] for dcm_slice in dcm_slices])
    slicespacing = np.mean(spacings)
    # All slices will have the same in-plane shape
    shape = (int(dcm_slices[0][0].Columns), int(dcm_slices[0][0].Rows))
    nslices = len(dcm_slices)
    # Final 3D array will be N_Slices x Columns x Rows
    shape = (nslices, *shape)
    img = np.empty(shape, dtype='float32')
    for idx, (dcm, _) in enumerate(dcm_slices):
        # Rescale and shift in order to get accurate pixel values
        slope = float(dcm.RescaleSlope)
        intercept = float(dcm.RescaleIntercept)
        img[idx, ...] = dcm.pixel_array.astype('float32')*slope + intercept
    # Calculate size of a voxel in mm
    pixelspacing = tuple(float(spac) for spac in dcm_slices[0][0].PixelSpacing)
    voxelspacing = (slicespacing, *pixelspacing)
    print('done.')
    return img, voxelspacing


def thru_plane_position(dcm):
    """Get spatial coordinate of image origin whose axis is perpendicular to image plane."""
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    position = tuple((float(p) for p in dcm.ImagePositionPatient))
    rowvec, colvec = orientation[:3], orientation[3:]
    normal_vector = np.cross(rowvec, colvec)
    slice_pos = np.dot(position, normal_vector)
    return slice_pos
