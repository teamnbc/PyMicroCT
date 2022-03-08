import cv2, os, math, operator, sys, time
import numpy as np

def imRescale2uint8(im):
    '''Rescale and convert to 8 bit ([min max] to [0 255])'''
    imOffset = im - np.min(im)
    imOut = 255 * (imOffset / np.max(imOffset))
    return(imOut.astype('uint8'))

def imLevels(im,low,high):
    '''Apply 2 thresholds: values < low are set to low, values > high are set to high'''
    ilow = np.where(im<=low) # Indices of pixels < low
    ihigh = np.where(im >= high)
    for i in list(zip(ilow[0],ilow[1])):
        im[i]=low
    for i in list(zip(ihigh[0], ihigh[1])):
        im[i]=high
    return(im)

def customResize(im,factor):
    '''Resize image using non integer factor'''
    imResized=cv2.resize(im, tuple(map(math.floor, tuple(factor * x for x in im.shape[0:2][::-1]))), interpolation=cv2.INTER_AREA)
    truefactor=tuple(map(operator.truediv, imResized.shape[0:2], im.shape[0:2]))[0]
    return imResized, truefactor

def imShowMe(im,msg='image'):
    cv2.imshow(msg, im)
    while True:
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
            cv2.destroyWindow(msg)
            break
        time.sleep(0.01)  # Slow down while loop to reduce CPU usage
