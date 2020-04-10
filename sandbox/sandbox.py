import numpy as np
import cv2, math, os
from scipy.interpolate import interp1d
import utilities as utils

# Goal: to delineate a polygon along the spine, which will be used as a mask to compute a side projection (voxels out of mask = zero).
# Rationale: obtain a side view of each vertebral body centered on its medial axis, with foramen, corpus and spinous process visible.
# Analysis steps are described below:
# 1. "Reference points" are drawn by the user.
# 2. A spline is interpolated between the reference points.
# 3. For each reference point, the coordinates of a pair of "side points" (A and B) is calculated such that :
#    - the line AB is perpendicular to the spline direction at the reference point.
#    - distance AB decreases linearly as we move along the spine towards its caudal aspect.
# 4. Calculate splines joining side points and define final polygon.

# Load original grayscale image and convert to RGB (so user can draw colored lines and circles)
img = cv2.merge(((cv2.imread('PROJ_top.png',0),)*3))
factor = 3
imgrs, resizefactor = utils.customResize(img, 3) # Will work on resized (larger) image
# Image coordinates system:
# Pixel numbers in original 512 x 512 image:
# **********************************
# * (0,0) → increasing x → (511,0) *
# *      ↓                         *
# * increasing y                   *
# *      ↓                         *
# * (0,511)              (511,511) *
# **********************************
# Convention: head is toward top of image, tail is toward bottom of image.
# Define how distance D between side points changes as we move along the spine towards the tail
# D = 30-(25/511)*y, meaning that D = 30 for y = 0 (top of image) and D = 5 for y = 511 (bottom of image)
# Reference points are defined as [[x1,y1],[x2,y2],...,[xn,yn]]
# Note that the points are not listed in order (ascending/descending values of x or y)
# This means that the user does not have to worry about selecting reference points in a specific order along the spine.
pts = [[240,100],[250,500],[260,300],[245,5],[240,400],[230,200]] # Reference points (list)
pts[:] = [[int(resizefactor*j) for j in i] for i in pts]
ptsArr = np.array(pts); ptsArrSorted = ptsArr[np.argsort(ptsArr[:, 1])] # Reference points sorted by ascending y value (ordered from head to tail)
# Compute spline joining reference points
x, y = np.array(ptsArrSorted)[:, 0], np.array(ptsArrSorted)[:, 1]
f = interp1d(y, x, kind='cubic') # Compute spline function
nPts = math.floor(np.ptp(y) * 100 / (512 * int(resizefactor))) # Goal: describe spline with 100 points (npts = 100).
yNew = np.linspace(np.min(y), np.max(y), num=nPts, endpoint=True) # Evenly spaced points along y axis
ptsNew = np.vstack((np.array(f(yNew)), yNew)).T # Coordinates of spline points
ptsNewInt = np.floor(ptsNew).astype('int16') # Same but integer values
for i in range(ptsNewInt.shape[0] - 1): cv2.line(imgrs, tuple(ptsNewInt[i, :]), tuple(ptsNewInt[i + 1, :]), (0, 255, 255), 2)
# Given n reference points defined by the user (used to compute the spline above)...
# ...compute normalized vectors describing direction of spline (dv: direction vector) at each reference point.
dv=utils.normv(np.vstack((np.array(np.append(f(y[:-1]+1)-x[:-1], -(f(y[-1]-1)-x[-1]))),[1 for i in range(0,len(y))])).T)
for i in range(dv.shape[0]): cv2.line(imgrs, tuple(ptsArrSorted[i]), tuple(np.floor(ptsArrSorted[i]+50*dv[i]).astype('int16')), (255, 0, 255), 2)
# Compute side points A and B (see description at begining of file); A is on the left, B is on the right
dvA, dvB = np.vstack((-dv[:,1],dv[:,0])).T, np.vstack((dv[:,1],-dv[:,0])).T # Simple 90 deg rotation
sideA, sideB = dvA.copy(), dvB.copy()
for i in range(ptsArrSorted.shape[0]): # Apply scaling factor such that D (distance between side points) decreases when moving toward the tail
    sideA[i, :] = ptsArrSorted[i] + (30 - (20 / (int(resizefactor*511))) * ptsArrSorted[i, 1]) * dvA[i]
    sideB[i, :] = ptsArrSorted[i] + (30 - (20 / (int(resizefactor*511))) * ptsArrSorted[i, 1]) * dvB[i]
sideAInt, sideBInt = np.floor(sideA).astype('int16'), np.floor(sideB).astype('int16') # For plotting only
for i in range(dv.shape[0]): cv2.line(imgrs, tuple(sideAInt[i]), tuple(sideBInt[i]), (255, 0, 0), 2)
# Compute splines for right side points (same method as above)
x, y = np.array(sideA)[:, 0], np.array(sideA)[:, 1]
f = interp1d(y, x, kind='cubic')
nPts = math.floor(np.ptp(y) * 100 / (512 * int(resizefactor)))
yNew = np.linspace(np.min(y), np.max(y), num=nPts, endpoint=True)
sideANew = np.vstack((np.array(f(yNew)), yNew)).T
sideANewInt = np.floor(sideANew).astype('int16')
for i in range(sideANewInt.shape[0] - 1): cv2.line(imgrs, tuple(sideANewInt[i, :]), tuple(sideANewInt[i + 1, :]), (0, 255, 255), 2)
# Compute spline for left side points
x, y = np.array(sideB)[:, 0], np.array(sideB)[:, 1]
f = interp1d(y, x, kind='cubic')
nPts = math.floor(np.ptp(y) * 100 / (512 * int(resizefactor)))
yNew = np.linspace(np.min(y), np.max(y), num=nPts, endpoint=True)
sideBNew = np.vstack((np.array(f(yNew)), yNew)).T
sideBNewInt = np.floor(sideBNew).astype('int16')
for i in range(sideBNewInt.shape[0] - 1): cv2.line(imgrs, tuple(sideBNewInt[i, :]), tuple(sideBNewInt[i + 1, :]), (0, 255, 255), 2)
for i in range(len(pts)): cv2.circle(imgrs,tuple(ptsArrSorted[i]),5,(0,255,0),-1) # Draw circle on each reference point


cv2.imshow('image',imgrs)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save annotated image with original size
refPnts = np.round(ptsArrSorted/resizefactor).astype('int16')
np.savetxt('/home/ghyomm/Desktop/POC5/refPnts.txt',refPnts,'%i',',')
rightPnts = np.round(sideA/resizefactor).astype('int16')
np.savetxt('/home/ghyomm/Desktop/POC5/rightPnts.txt',rightPnts,'%i',',')
leftPnts = np.round(sideB/resizefactor).astype('int16')
np.savetxt('/home/ghyomm/Desktop/POC5/leftPnts.txt',leftPnts,'%i',',')
refPntsSpline = np.round(ptsNew/resizefactor).astype('int16')
np.savetxt('/home/ghyomm/Desktop/POC5/refPntsSpline.txt',refPntsSpline,'%i',',')
rightPntsSpline = np.round(sideANew/resizefactor).astype('int16')
np.savetxt('/home/ghyomm/Desktop/POC5/rightPntsSpline.txt',rightPntsSpline,'%i',',')
leftPntsSpline = np.round(sideBNew/resizefactor).astype('int16')
np.savetxt('/home/ghyomm/Desktop/POC5/leftPntsSpline.txt',leftPntsSpline,'%i',',')
for i in range(refPntsSpline.shape[0] - 1): cv2.line(img, tuple(refPntsSpline[i, :]), tuple(refPntsSpline[i + 1, :]),(0, 255, 255), 2)
for i in range(rightPntsSpline.shape[0] - 1): cv2.line(img, tuple(rightPntsSpline[i, :]), tuple(rightPntsSpline[i + 1, :]), (0, 255, 255), 2)
for i in range(leftPntsSpline.shape[0] - 1): cv2.line(img, tuple(leftPntsSpline[i, :]), tuple(leftPntsSpline[i + 1, :]), (0, 255, 255), 2)
for i in range(dv.shape[0]):
    cv2.line(img, tuple(refPnts[i]), tuple(np.floor(refPnts[i]+50*dv[i]).astype('int16')), (255, 0, 255), 2)
    cv2.line(img, tuple(rightPnts[i]), tuple(leftPnts[i]), (255, 0, 0), 2)
for i in range(refPnts.shape[0]): cv2.circle(img, tuple(refPnts[i]),3,(0, 255, 0),-1)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('/home/ghyomm/Desktop/POC5/img.png', img)