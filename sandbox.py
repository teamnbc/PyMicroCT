import os, cv2, pydicom, roi, time, pickle, sys, glob
import numpy as np
import utilities as utils
from datetime import date
from scipy.interpolate import interp1d
import symmetry as sym
from cython_code import cyutils
import pandas as pd
data = pydicom.dcmread('/mnt/data/DATA_SSPO/export_windows.dcm')
data.PixelSpacing[1]

arr=data.pixel_array
imOffset = arr + 32768
imOut = 255 * (imOffset / np.max(imOffset))
im_rear = np.amax(imOut, axis=2)  # Projection (rear view)
cv2.imwrite('/mnt/data/DATA_SSPO/test.png', im_rear)
