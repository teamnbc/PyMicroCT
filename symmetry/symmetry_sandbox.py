
'''
Prerequisite: check instructions in cython_code/setup.py

Note: reference frames in numpy and opencv2
[x,y] in numpy array:
x = row; y = col
***********************************
* [0,0] → increasing y → [0,ncol] *
*      ↓                          *
* increasing x                    *
*      ↓                          *
* [nrow,0]            [nrow,ncol] *
***********************************
[x,y] in opencv2:
x = index along width; y = index along height
************************************
* [0,0] → increasing x → [width,0] *
*      ↓                           *
* increasing y                     *
*      ↓                           *
* [0,height]        [width,height] *
************************************
'''

import cv2, os, math, time
import utilities as utils
import numpy as np
from cython_code import cyutils
from symmetry.sym import test_angle_offset

'''Test calculation of best symmetry axis'''
os.chdir('/home/ghyomm/DATA_MICROCT/SPINE/20201014_SPINE/CN724_35_Colonne_104238')
im = cv2.imread(os.path.join(os.getcwd(),'analysis','images','vertebrae','raw','00_S4.png'), cv2.IMREAD_GRAYSCALE)
refpt = np.array([100,100],dtype=np.uint16)
angle_range = np.array([-30,30],dtype=np.int16)
hoffset_range = np.array([-20,21],dtype=np.int16)
im_out = cyutils.compute_sym_axis(im,refpt,angle_range,hoffset_range)
max_coord = np.where(im_out == np.amax(im_out))
best_angle = int(max_coord[0] + angle_range[0])
best_offset = int(max_coord[1] + hoffset_range[0])
compute_angle_and_offset(im,best_angle,best_offset)

'''Illustration image describing calculation'''
im = cv2.imread(os.path.join(os.getcwd(),'symmetry','images','vertebra_example_1.png'), cv2.IMREAD_GRAYSCALE)
im = cv2.merge(((im,) * 3))  # Convert to RGB
refpt = np.array([100,100])  # Reference point, approximately at the refpt of vertebra
cv2.circle(im, tuple(refpt), 3, (0, 0, 255), -1)
hoffset = 20  # Horizontal offset
angle = 20  # Tilt angle
ylim = 70  # Upper and lower limits for drawing symmetry line

# new_refpt = refpt + np.array([hoffset, 0])
# cv2.circle(im, tuple(new_refpt), 3, (255, 255, 0), -1)
# # Top point
# yt = - ylim
# xt = - int(round(yt * math.tan(angle * math.pi / 180)))
# top_pt = new_refpt + np.array([xt, yt])
# cv2.circle(im, tuple(top_pt), 3, (255, 0, 0), -1)
# # Bottom point
# yb = ylim
# xb = - int(round(yb * math.tan(angle * math.pi / 180)))
# bottom_pt = new_refpt + np.array([xb, yb])
# cv2.circle(im, tuple(bottom_pt), 3, (0, 255, 255), -1)
# # Symmetry line
# cv2.line(im, tuple(top_pt), tuple(bottom_pt), (0, 255, 0), 1)
# # Vector defining symmetry line
# symvec = utils.norml(bottom_pt - new_refpt)
# # Take random point and compute projection onto symmetry line
# rnd_pt = np.array([70,70])
# proj_pt = new_refpt + np.round(symvec * np.sum((rnd_pt - new_refpt) * symvec)).astype('int')
# sym_pt = 2 * proj_pt - rnd_pt
# cv2.circle(im, tuple(rnd_pt), 3, (100, 100, 100), -1)
# # cv2.circle(im, tuple(proj_pt), 3, (255, 0, 255), -1)
# cv2.circle(im, tuple(sym_pt), 3, (255, 0, 255), -1)
# cv2.line(im, tuple(rnd_pt), tuple(sym_pt), (255, 0, 255), 1)
# im, fac = utils.customResize(im, 2)
# # cv2.imwrite('/home/ghyomm/Desktop/im.png',im)
# cv2.imshow('im',im)
# while True:
#     if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
#         cv2.destroyWindow('im')
#         break
#     time.sleep(0.01)  # Slow down while loop to reduce CPU usage
