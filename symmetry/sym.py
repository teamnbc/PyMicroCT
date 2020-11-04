
'''
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


import cv2, time, math, os
import utilities as utils
import numpy as np
from cython_code import cyutils


def define_symvec(angle):
    '''Compute vector giving direction of symmetry axis'''
    return(utils.norml(np.array([- math.tan(angle * math.pi / 180), 1])))


def sym_point(refpt,vec,pt):
    '''Compute coordinates of symmetrical point'''
    proj_pt = refpt + vec * np.sum((pt - refpt) * vec)
    return(np.trunc(2 * proj_pt - pt).astype('int'))


def compute_angle_and_offset(im,angle,hoffset,refpt=np.array([100,100],dtype=np.uint16)):
    '''Test overlap btw original and mirror image using...
    ... a symmetry axis defined by a specific angle and offset '''
    '''
    Rationale: move the reference point along horizontal line
    to the left and right (over defined interval)
    For each position of the reference point, draw line passing through the reference point
    and change tilt angle of the line
    '''
    new_refpt = refpt + np.array([hoffset, 0],dtype=np.uint16)
    im_sym, counts = cyutils.compute_mirror_image(im,new_refpt,vec=define_symvec(angle))
    im_rgb = cv2.merge((im,np.zeros_like(im).astype('uint8'),im_sym))
    # Draw symmetry line
    ylim = 100
    yt = - ylim
    xt = - int(round(yt * math.tan(angle * math.pi / 180)))
    top_pt = new_refpt + np.array([xt, yt])
    yb = ylim
    xb = - int(round(yb * math.tan(angle * math.pi / 180)))
    bottom_pt = new_refpt + np.array([xb, yb])
    cv2.line(im_rgb, tuple(top_pt), tuple(bottom_pt), (0, 255, 255), 1)
    cv2.circle(im_rgb, tuple(refpt), 2, (0, 255, 0), -1)
    cv2.circle(im_rgb, tuple(new_refpt), 2, (0, 255, 255), -1)
    cv2.putText(im_rgb, str(counts) + ' overlapping pixels', (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.putText(im_rgb, str(angle) + ' deg', (5, np.shape(im)[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.putText(im_rgb, 'hoffset = ' + str(hoffset) + ' px', (np.shape(im)[1]-130, np.shape(im)[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
    im_rgb_resized, fac = utils.customResize(im_rgb, 2)
    return im_rgb_resized
    # cv2.imshow('im',im_rgb_resized)
    # while True:
    #     if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
    #         cv2.destroyWindow('im')
    #         break
    #         time.sleep(0.01)  # Slow down while loop to reduce CPU usage
