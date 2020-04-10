##############################################
# customROI.py                               #
# part of PyMicroCT project                  #
# Classes to draw custom ROIs on images...   #
# ... and perform simple operations on them. #
##############################################


import numpy as np
import cv2, time, math
import utilities as utils
from scipy.interpolate import interp1d


# imsrc (source image, displayed), msg (window title), fact (scaling factor)
# horiz: draw horizontal line on rear projection
# rear: draw contour of mouse on rear projection (to exclude tube and have cleaner max projections)

# Main class for steps 3 to 5 in main.py
class CustROI1:
    def __init__(self, imsrc_horiz, msg_horiz, fact_horiz,
                 imsrc_rear, msg_rear, fact_rear,
                 imsrc_side, msg_side, fact_side):
        self.horiz = DrawLine(imsrc_horiz, msg_horiz, fact_horiz)  # Draw line underneath mouse
        self.rear = DrawPoly(imsrc_rear, msg_rear, fact_rear)  # Draw polygon around mouse on rear view
        self.side = DrawPoly(imsrc_side, msg_side, fact_side)  # Draw polygon around spine on side view


# Main class for step 6 in main.py
class CustROI2:
    def __init__(self, imsrc_top, msg_top, fact_top):
        self.top = DrawSpineAxis(imsrc_top, msg_top, fact_top)


# Main class for step 7 in main.py
class CustROI3:
    def __init__(self, imsrc_side, msg_side, fact_side, L6_position):
        self.side = DrawVertebraeLimits(imsrc_side, msg_side, fact_side, L6_position)


class DrawLine:
    """Class used to draw a line underneath the mouse to calculate tilt angle"""
    def __init__(self, im, msg, fact):
        self.im = im
        self.fact = fact
        self.msg = msg
        self.pts = []
        self.angle = None
        self.closest = None
        self.closest_last = None
        self.im_resized, self.resize_factor = utils.customResize(self.im, self.fact)
        self.im_copy = self.im_resized.copy()

    def DrawHorizLine(self):
        cv2.imshow(self.msg, self.im_resized)  # Initialize window with proper title
        cv2.setMouseCallback(self.msg, self.callback)  # Bind window to the callback
        while True:
            cv2.imshow(self.msg, self.im_resized)  # Update display to show lines
            for i in range(len(self.pts)):
                cv2.circle(self.im_resized, tuple(self.pts[i]), 5, (0, 255, 0), -1)
            if self.closest is not None:
                cv2.circle(self.im_resized, tuple(self.closest), 20, (255, 255, 0), 1)
            if len(self.pts) > 1:
                for i in range(len(self.pts) - 1):
                    cv2.line(self.im_resized, tuple(self.pts[i]), tuple(self.pts[i + 1]), (0, 255, 0), 2)
                self.angle = round(math.atan((self.pts[1][1] - self.pts[0][1]) /
                                             (self.pts[1][0] - self.pts[0][0])) * (180 / math.pi), 2)
                xtext = int(math.floor((self.pts[0][0] + self.pts[1][0])/2))
                ytext = int(math.floor((self.pts[0][1] + self.pts[1][1])/2))
                cv2.putText(self.im_resized, str(self.angle)+" deg", (xtext, ytext),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
                if len(self.pts) == 2:  # Means user has drawn line
                    self.pts = np.round(np.array(self.pts) / self.resize_factor).astype(int)
                    # Annotate original image
                    for i in range(len(self.pts)):
                        cv2.circle(self.im, tuple(self.pts[i]), 5, (0, 255, 0), -1)
                    for i in range(len(self.pts) - 1):
                        cv2.line(self.im, tuple(self.pts[i]), tuple(self.pts[i + 1]), (0, 255, 0), 2)
                    xtext = int(math.floor((self.pts[0][0] + self.pts[1][0]) / 2))
                    ytext = int(math.floor((self.pts[0][1] + self.pts[1][1]) / 2))
                    cv2.putText(self.im, str(self.angle) + " deg", (xtext, ytext),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)
                else:
                    self.pts = []
                cv2.destroyWindow(self.msg)
                break  # Return to main function
            time.sleep(0.01)  # Slow down while loop to reduce CPU usage

    def callback(self, event, x, y, flags, params):
        """Respond to mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.pts) < 2:
            self.pts.append([x, y])
        if event == cv2.EVENT_MBUTTONDOWN and (len(self.pts) >= 1):
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts),2),1)
            del self.pts[diff.argmin()]
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts), 2), 1)
            self.closest = self.pts[diff.argmin()]
            self.im_resized = self.im_copy.copy()
            self.closest_last = self.closest
            self.something_changed = True
        if event == cv2.EVENT_MOUSEMOVE and (len(self.pts) >= 1):
            # Find reference point closest to latest mouse position (= closest reference point)
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts), 2), 1)
            self.closest = self.pts[diff.argmin()]
            # Update coordinates of closest reference point
            if self.closest_last != self.closest:
                self.im_resized = self.im_copy.copy()
                self.closest_last = self.closest
                self.something_changed = True


class DrawPoly:
    """Class used to draw a polygon on rear and side projections"""
    def __init__(self, im, msg, fact):
        self.im = im  # The image on which the ROI is drawn
        self.fact = fact  # Scaling factor (so the user can point & click on larger image)
        self.msg = msg  # Window title
        self.immask = np.zeros(self.im.shape[0:2], np.uint8)  # Prepare mask
        self.pts = []  # Coordinates of points will be stored here
        self.something_changed = True
        self.closest = None
        self.closest_last = None
        self.im_resized, self.resize_factor = utils.customResize(self.im, self.fact)
        self.im_copy = self.im_resized.copy()
        self.mag_width = 20


    def drawpolygon(self):

        mag_left = [[self.mag_width, 0], [self.mag_width, 511]]
        mag_right = [[511 - self.mag_width, 0], [511 - self.mag_width, 511]]
        cv2.line(self.im_resized, tuple([math.floor(self.resize_factor * i) for i in mag_left[0]]),
                 tuple([math.floor(self.resize_factor * i) for i in mag_left[1]]), (200, 200, 200), 1)
        cv2.line(self.im_resized, tuple([math.floor(self.resize_factor * i) for i in mag_right[0]]),
                 tuple([math.floor(self.resize_factor * i) for i in mag_right[1]]), (200, 200, 200), 1)
        cv2.imshow(self.msg, self.im_resized)  # Initialize window with proper title
        cv2.setMouseCallback(self.msg, self.callback)  # Invoke callback function to listen to mouse events
        while True:
            if self.something_changed:
                if len(self.pts) > 1:
                    for i in range(len(self.pts) - 1):  # Draw line
                        cv2.line(self.im_resized, tuple(self.pts[i]), tuple(self.pts[i + 1]), (0, 255, 0), 2)
                if len(self.pts) > 0:
                    for i in range(len(self.pts)):  # Draw circle on each reference point
                        cv2.circle(self.im_resized, tuple(self.pts[i]), 5, (0, 255, 0), -1)
                    if self.closest is not None:
                        cv2.circle(self.im_resized, tuple(self.closest), 20, (255, 255, 0), 3)
                cv2.line(self.im_resized, tuple([math.floor(self.resize_factor * i) for i in mag_left[0]]),
                         tuple([math.floor(self.resize_factor * i) for i in mag_left[1]]), (200, 200, 200), 1)
                cv2.line(self.im_resized, tuple([math.floor(self.resize_factor * i) for i in mag_right[0]]),
                         tuple([math.floor(self.resize_factor * i) for i in mag_right[1]]), (200, 200, 200), 1)
                cv2.imshow(self.msg, self.im_resized)  # Update display
                self.something_changed = False
            if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
                if len(self.pts) > 2:  # Update mask
                    self.pts = np.round(np.array(self.pts) / self.resize_factor).astype(int)
                    for i in range(len(self.pts) - 1):  # Draw line
                        cv2.line(self.im, tuple(self.pts[i]), tuple(self.pts[i + 1]), (0, 255, 0), 2)
                    cv2.fillPoly(self.immask, np.array([self.pts]), (255, 255, 255), 1)
                else:
                    self.pts = []
                cv2.destroyWindow(self.msg)
                break  # Return to main function
            time.sleep(0.01)  # Slow down while loop to reduce CPU usage

    def callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < math.floor(self.resize_factor*self.mag_width):
                x = 0
            if x > self.im_resized.shape[0] - math.floor(self.resize_factor*self.mag_width):
                x = self.im_resized.shape[0]
            self.pts.append([x, y])
            self.im_resized = self.im_copy.copy()
            self.something_changed = True
        if event == cv2.EVENT_MBUTTONDOWN and (len(self.pts) > 0):
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts),2),1)
            del self.pts[diff.argmin()]
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts), 2), 1)
            self.closest = self.pts[diff.argmin()]
            self.im_resized = self.im_copy.copy()
            self.closest_last = self.closest
            self.something_changed = True
        if event == cv2.EVENT_MOUSEMOVE and (len(self.pts) > 0):
            # Find reference point closest to latest mouse position (= closest reference point)
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts), 2), 1)
            self.closest = self.pts[diff.argmin()]
            # Update coordinates of closest reference point
            if self.closest_last != self.closest:
                self.im_resized = self.im_copy.copy()
                self.closest_last = self.closest
                self.something_changed = True


# class DrawL6:
#     """Class used to draw a point at the level of L6 vertebra"""
#     def __init__(self, im, msg, fact):
#         self.im = im
#         self.fact = fact
#         self.msg = msg
#         self.pts = []
#         self.im_resized, self.resize_factor = utils.customResize(self.im, self.fact)
#         self.im_copy = self.im_resized.copy()


class DrawSpineAxis:
    """
    Class used to draw main spine axis on top projection
    Goal: to delineate a polygon along the spine...
    ...which will be used as a mask to compute a side projection (voxels out of mask = zero).
    Rationale: obtain a side view of each vertebral body centered on its medial axis...
    ...with foramen, corpus and spinous process visible.
    Analysis steps are described below:
    1. "Reference points" are drawn by the user.
    2. A spline is interpolated between the reference points.
    3. For each reference point, the coordinates of a pair of "side points" (A and B) is calculated such that :
       - the line AB is perpendicular to the spline direction at the reference point.
       - distance AB decreases linearly as we move along the spine towards its caudal aspect.
    4. Calculate splines joining side points and define final polygon.
    """
    def __init__(self, im, msg, fact):
        self.im = im
        self.fact = fact
        self.msg = msg
        self.immask = np.zeros(self.im.shape[0:2], np.uint8)  # Prepare mask
        self.pts = []  # Reference points (list)
        self.something_changed = True
        self.closest = None
        self.closest_last = None
        self.im_resized, self.resize_factor = utils.customResize(self.im, self.fact)
        self.im_copy = self.im_resized.copy()
        self.mag_width = 20
        self.L6_pos = None  # Position of vertebrae L6
        self.first_call = True

    def DrawL6(self):
        """
        Indicate position of L6 vertebra.
        The user simply has to click onto the L6 vertebra...
        ... and the y position is saved in self.L6_pos.
        """
        cv2.imshow(self.msg, self.im_resized)  # Initialize window with title
        cv2.setMouseCallback(self.msg, self.callback1)  # Bind window to the callback
        while True:
            cv2.imshow(self.msg, self.im_resized)  # Update display
            if(self.first_call):
                cv2.putText(self.im_resized, "Please click onto L6 vertebra",
                    (50, int(round(100 * self.resize_factor))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType = cv2.LINE_AA)
                cv2.putText(self.im_resized, "and then press 'q'.",
                    (50, int(round(112 * self.resize_factor))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType = cv2.LINE_AA)
                self.first_call = False
            if(len(self.pts)>0): # selt.pts contains one point max
                self.L6_pos = int(round(self.pts[1] / self.resize_factor))
                cv2.circle(self.im_resized, tuple(self.pts), 5, (0, 255, 0), -1)
                cv2.circle(self.im_resized, tuple(self.pts), 50, (0, 255, 0), 2)
                cv2.line(self.im_resized, tuple([0, self.pts[1]]),
                    tuple([math.floor(self.resize_factor * 511), self.pts[1]]),
                    (0, 255, 0), 2)  # Line showing position of L6
                cv2.putText(self.im_resized, "Position of L6 vertebra = " + str(self.L6_pos),
                    (20, self.pts[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
                if len(self.pts) > 0:  # Means user has drawn point
                    self.L6_pos = int(round(self.pts[1] / self.resize_factor))
                else:
                    self.L6_pos = []
                self.im_resized = self.im_copy.copy()
                cv2.destroyWindow(self.msg)
                break  # Return to main function
            time.sleep(0.01)  # Slow down while loop to reduce CPU usage

    def DrawSpline(self):
        """
        Draw spline along spine axis.
        This class is coupled to callback2 (for handling mouse events).
        Pixel numbers in original 512 x 512 image:
        **********************************
        * (0,0) → increasing x → (511,0) *
        *      ↓                         *
        * increasing y                   *
        *      ↓                         *
        * (0,511)              (511,511) *
        **********************************
        Convention: head is toward top of image, tail is toward bottom of image.
        Define how distance D between side points changes as we move along the spine towards the tail
        D = 30-(25/511)*y, meaning that D = 30 for y = 0 (top of image) and D = 5 for y = 511 (bottom of image)
        Reference points are defined as [[x1,y1],[x2,y2],...,[xn,yn]]
        Note that the points are not listed in order (ascending/descending values of x or y)
        This means that the user does not have to worry about selecting reference points...
        ...in a specific order along the spine.
        Initialize up and bottom limits for "magnetism" of extreme points
        """
        self.first_call = True # Because was used by DrawL6 just before.
        self.pts = [] # Resetting self.pts (used by DrawL6 just before).
        mag_top = [[0, self.mag_width], [511, self.mag_width]]
        mag_bottom = [[0, 511 - self.mag_width], [511, 511 - self.mag_width]]
        cv2.line(self.im_resized, tuple([math.floor(self.resize_factor * i) for i in mag_top[0]]),
                 tuple([math.floor(self.resize_factor * i) for i in mag_top[1]]), (200, 200, 200), 1)
        cv2.line(self.im_resized, tuple([math.floor(self.resize_factor * i) for i in mag_bottom[0]]),
                 tuple([math.floor(self.resize_factor * i) for i in mag_bottom[1]]), (200, 200, 200), 1)
        cv2.imshow(self.msg, self.im_resized)  # Initialize window with proper title
        cv2.setMouseCallback(self.msg, self.callback2)  # Bind window to the callback
        while True:
            if self.first_call:
                cv2.putText(self.im_resized, "Please draw spine axis",
                    (50, int(round(100 * self.resize_factor))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType = cv2.LINE_AA)
                cv2.putText(self.im_resized, "and then press 'q'.",
                    (50, int(round(112 * self.resize_factor))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType = cv2.LINE_AA)
                cv2.imshow(self.msg, self.im_resized)
                self.first_call = False
            if self.something_changed:
                # Upper and lower lines (for magnetism of extreme points)
                cv2.line(self.im_resized, tuple([math.floor(self.resize_factor * i) for i in mag_top[0]]),
                         tuple([math.floor(self.resize_factor * i) for i in mag_top[1]]), (200, 200, 200), 1)
                cv2.line(self.im_resized, tuple([math.floor(self.resize_factor * i) for i in mag_bottom[0]]),
                         tuple([math.floor(self.resize_factor * i) for i in mag_bottom[1]]), (200, 200, 200), 1)
                if len(self.pts) > 0:
                    # Reference points sorted by ascending y value (ordered from head to tail)
                    pts_arr = np.array(self.pts)
                    pts_arr_sorted = pts_arr[np.argsort(pts_arr[:, 1])]
                    if len(self.pts) <= 3:
                        for i in range(len(self.pts)):  # Draw circle on each reference point
                            cv2.circle(self.im_resized, tuple(pts_arr_sorted[i]), 5, (0, 255, 0), -1)
                    if len(self.pts) > 3:
                        # Compute spline joining reference points
                        x, y = np.array(pts_arr_sorted)[:, 0], np.array(pts_arr_sorted)[:, 1]
                        f = interp1d(y, x, kind='cubic')  # Compute spline function
                        # Goal: describe spline with 100 points (npts = 100).
                        npts = math.floor(np.ptp(y) * 100 / (512 * int(self.resize_factor)))
                        # Evenly spaced points along y axis
                        ynew = np.linspace(np.min(y), np.max(y), num=npts, endpoint=True)
                        ptsnew = np.vstack((np.array(f(ynew)), ynew)).T  # Coordinates of spline points
                        ptsnewint = np.floor(ptsnew).astype('int16')  # Same but integer values
                        for i in range(ptsnewint.shape[0] - 1):
                            cv2.line(self.im_resized, tuple(ptsnewint[i, :]),
                                     tuple(ptsnewint[i + 1, :]), (0, 255, 255), 1)
                        # Given n reference points defined by the user (used to compute the spline above)...
                        # ...compute normalized vectors describing direction of spline (dv: direction vector) at each reference point.
                        dv = utils.Norml(np.vstack((np.array(np.append(f(y[:-1] + 1) - x[:-1], -(f(y[-1] - 1) - x[-1]))),
                                                    [1 for i in range(0, len(y))])).T)
                        for i in range(dv.shape[0]):
                            cv2.line(self.im_resized,
                                     tuple(pts_arr_sorted[i]),
                                     tuple(np.floor(pts_arr_sorted[i] + 50 * dv[i]).astype('int16')), (255, 0, 255), 2)
                        # Compute side points A and B (see description at begining of file); A is on the left, B is on the right
                        dva, dvb = np.vstack((-dv[:, 1], dv[:, 0])).T, np.vstack((dv[:, 1], -dv[:, 0])).T  # Simple 90 deg rotation
                        side_a, side_b = dva.copy(), dvb.copy()
                        # Apply scaling factor such that D (distance between side points) decreases when moving toward the tail
                        for i in range(pts_arr_sorted.shape[0]):
                            # Here: D = 30 at top of image and D = 10 at bottom of image
                            side_a[i, :] = pts_arr_sorted[i] + (10 - (5 / (int(self.resize_factor * 511))) * pts_arr_sorted[i, 1]) * dva[i]
                            side_b[i, :] = pts_arr_sorted[i] + (10 - (5 / (int(self.resize_factor * 511))) * pts_arr_sorted[i, 1]) * dvb[i]
                        side_a_int, side_b_int = np.floor(side_a).astype('int16'), np.floor(side_b).astype('int16')  # For plotting only
                        for i in range(dv.shape[0]):
                            cv2.line(self.im_resized, tuple(side_a_int[i]), tuple(side_b_int[i]), (255, 0, 0), 1)
                        # Compute splines for right side points (same method as above)
                        x, y = np.array(side_a)[:, 0], np.array(side_a)[:, 1]
                        f = interp1d(y, x, kind='cubic')
                        npts = math.floor(np.ptp(y) * 100 / (512 * int(self.resize_factor)))
                        ynew = np.linspace(np.min(y), np.max(y), num=npts, endpoint=True)
                        side_a_new = np.vstack((np.array(f(ynew)), ynew)).T
                        side_a_new_int = np.floor(side_a_new).astype('int16')
                        for i in range(side_a_new_int.shape[0] - 1):
                            cv2.line(self.im_resized, tuple(side_a_new_int[i, :]), tuple(side_a_new_int[i + 1, :]),
                                     (0, 255, 255), 1)
                        # Compute spline for left side points
                        x, y = np.array(side_b)[:, 0], np.array(side_b)[:, 1]
                        f = interp1d(y, x, kind='cubic')
                        npts = math.floor(np.ptp(y) * 100 / (512 * int(self.resize_factor)))
                        ynew = np.linspace(np.min(y), np.max(y), num=npts, endpoint=True)
                        side_b_new = np.vstack((np.array(f(ynew)), ynew)).T
                        side_b_new_int = np.floor(side_b_new).astype('int16')
                        for i in range(side_b_new_int.shape[0] - 1):
                            cv2.line(self.im_resized, tuple(side_b_new_int[i, :]), tuple(side_b_new_int[i + 1, :]),
                                     (0, 255, 255), 1)
                        for i in range(len(self.pts)):  # Draw circle on each reference point
                            cv2.circle(self.im_resized, tuple(pts_arr_sorted[i]), 5, (0, 255, 0), -1)
                    if self.closest is not None:
                        cv2.circle(self.im_resized, tuple(self.closest), 20, (255, 255, 0), 3)
                    cv2.imshow(self.msg, self.im_resized)  # Update display
                self.something_changed = False

            if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
                if len(self.pts) > 0:
                    if len(self.pts) > 3:
                        pts = np.round(np.array(self.pts) / self.resize_factor).astype(int)
                        pts_arr = np.array(pts)
                        pts_arr_sorted = pts_arr[np.argsort(pts_arr[:, 1])]
                        # Compute spline joining reference points
                        x, y = np.array(pts_arr_sorted)[:, 0], np.array(pts_arr_sorted)[:, 1]
                        f = interp1d(y, x, kind='cubic')  # Compute spline function
                        # Goal: describe spline with 100 points (npts = 100).
                        npts = math.floor(np.ptp(y) * 100 / 512)
                        # Evenly spaced points along y axis
                        ynew = np.linspace(np.min(y), np.max(y), num=npts, endpoint=True)
                        ptsnew = np.vstack((np.array(f(ynew)), ynew)).T  # Coordinates of spline points
                        ptsnewint = np.floor(ptsnew).astype('int16')  # Same but integer values
                        for i in range(ptsnewint.shape[0] - 1):
                            cv2.line(self.im, tuple(ptsnewint[i, :]),
                                     tuple(ptsnewint[i + 1, :]), (0, 255, 255), 1)
                        # Given n reference points defined by the user (used to compute the spline above)...
                        # ...compute normalized vectors describing direction of spline (dv: direction vector) at each reference point.
                        dv = utils.Norml(np.vstack((np.array(np.append(f(y[:-1] + 1) - x[:-1], -(f(y[-1] - 1) - x[-1]))),
                                                    [1 for i in range(0, len(y))])).T)
                        # Compute side points A and B (see description at begining of file); A is on the left, B is on the right
                        dva, dvb = np.vstack((-dv[:, 1], dv[:, 0])).T, np.vstack((dv[:, 1], -dv[:, 0])).T  # Simple 90 deg rotation
                        side_a, side_b = dva.copy(), dvb.copy()
                        # Apply scaling factor such that D (distance between side points) decreases when moving toward the tail
                        for i in range(pts_arr_sorted.shape[0]):
                            side_a[i, :] = pts_arr_sorted[i] + (8 - (8 / 511) * pts_arr_sorted[i, 1]) * dva[i]
                            side_b[i, :] = pts_arr_sorted[i] + (8 - (8 / 511) * pts_arr_sorted[i, 1]) * dvb[i]
                        # Compute splines for right side points (same method as above)
                        x, y = np.array(side_a)[:, 0], np.array(side_a)[:, 1]
                        f = interp1d(y, x, kind='cubic')
                        npts = math.floor(np.ptp(y) * 100 / 512)
                        ynew = np.linspace(np.min(y), np.max(y), num=npts, endpoint=True)
                        side_a_new = np.vstack((np.array(f(ynew)), ynew)).T
                        side_a_new_int = np.floor(side_a_new).astype('int16')
                        for i in range(side_a_new_int.shape[0] - 1):
                            cv2.line(self.im, tuple(side_a_new_int[i, :]), tuple(side_a_new_int[i + 1, :]),
                                     (0, 255, 255), 1)
                        # Compute spline for left side points
                        x, y = np.array(side_b)[:, 0], np.array(side_b)[:, 1]
                        f = interp1d(y, x, kind='cubic')
                        npts = math.floor(np.ptp(y) * 100 / 512)
                        ynew = np.linspace(np.min(y), np.max(y), num=npts, endpoint=True)
                        side_b_new = np.vstack((np.array(f(ynew)), ynew)).T
                        side_b_new_int = np.floor(side_b_new).astype('int16')
                        for i in range(side_b_new_int.shape[0] - 1):
                            cv2.line(self.im, tuple(side_b_new_int[i, :]), tuple(side_b_new_int[i + 1, :]),
                                     (0, 255, 255), 1)
                        for i in range(len(pts)):  # Draw circle on each reference point
                            cv2.circle(self.im, tuple(pts_arr_sorted[i]), 2, (0, 255, 0), -1)
                        # Find points of left and right splines which are outside image
                        side_a_new_int_sorted = side_a_new_int[np.argsort(side_a_new_int[:, 1])]
                        side_a_new_int_sorted[side_a_new_int_sorted < 0] = 0
                        side_a_new_int_sorted[side_a_new_int_sorted > 511] = 511
                        side_b_new_int_sorted = side_b_new_int[np.argsort(-side_b_new_int[:, 1])]
                        side_b_new_int_sorted[side_b_new_int_sorted < 0] = 0
                        side_b_new_int_sorted[side_b_new_int_sorted > 511] = 511
                        ll = np.vstack((side_a_new_int_sorted,side_b_new_int_sorted)).tolist()
                        cv2.fillPoly(self.immask, np.array([ll]), (255, 255, 255), 1)
                else:
                    self.pts = []
                cv2.destroyWindow(self.msg)
                break  # Return to main function
            time.sleep(0.01)  # Slow down while loop to reduce CPU usage

    def callback1(self, event, x, y, flags, params):
        """Respond to mouse events for class DrawL6"""
        # Here very simple: self.pts has just one point
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts = [x, y]
            self.im_resized = self.im_copy.copy()

    def callback2(self, event, x, y, flags, params):
        """Respond to mouse events for class DrawSpline"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if y < math.floor(self.resize_factor*self.mag_width):
                y = 0
            if y > self.im_resized.shape[0] - math.floor(self.resize_factor*self.mag_width):
                y = self.im_resized.shape[0]
            self.pts.append([x, y])
            self.closest = [x, y] # Update coordinates of closest reference point
            self.closest_last = self.closest
            self.im_resized = self.im_copy.copy()
            self.something_changed = True
        if event == cv2.EVENT_MBUTTONDOWN and (len(self.pts) > 0):
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts),2),1)
            del self.pts[diff.argmin()]
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts), 2), 1)
            self.closest = self.pts[diff.argmin()]
            self.im_resized = self.im_copy.copy()
            self.closest_last = self.closest
            self.something_changed = True
        if event == cv2.EVENT_MOUSEMOVE and (len(self.pts) > 0):
            # Find reference point closest to latest mouse position (= closest reference point)
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts), 2), 1)
            self.closest = self.pts[diff.argmin()]
            # Update coordinates of closest reference point
            if self.closest_last != self.closest:
                self.im_resized = self.im_copy.copy()
                self.closest_last = self.closest
                self.something_changed = True


class DrawVertebraeLimits:
    """Class used to draw limits of vertebrae on side projection"""
    def __init__(self, im, msg, fact, L6):
        self.im = im  # The image on which the ROI is drawn
        self.fact = fact  # Scaling factor (so the user can point & click on larger image)
        self.msg = msg  # Window title
        self.pts = []  # Coordinates of points will be stored here
        self.midpnt = []  # Midpoints (center of vertebrae)
        self.something_changed = True
        self.closest = None
        self.closest_last = None
        self.im_resized, self.resize_factor = utils.customResize(self.im, self.fact)
        self.im_copy = self.im_resized.copy()
        self.L6 = L6
        # List of vertebrae:
        self.V_ID_list = [e + str(i + 1) for i, e in enumerate(['Ce'] * 7)] + \
                         [e + str(i + 1) for i, e in enumerate(['T'] * 13)] + \
                         [e + str(i + 1) for i, e in enumerate(['L'] * 6)] + \
                         [e + str(i + 1) for i, e in enumerate(['S'] * 4)] + \
                         [e + str(i + 1) for i, e in enumerate(['Ca'] * 20)]
        self.V_ID = [] # Where the names of annotated vertebrae will be stored

    def DrawVertebrae(self):
        """Draw limits of vertebrae on side projection"""
        cv2.line(self.im_resized, tuple([math.floor(self.resize_factor * self.L6), 0]),
                 tuple([math.floor(self.resize_factor * self.L6), math.floor(self.resize_factor * 511)]), (0, 0, 255), 1)
        cv2.putText(self.im_resized, "L6", (math.floor(self.resize_factor * self.L6), math.floor(self.resize_factor * 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.imshow(self.msg, self.im_resized)  # Initialize window with proper title
        cv2.setMouseCallback(self.msg, self.callback)  # Bind window to the callback
        while True:
            if self.something_changed: # Don't bother updating window if nothing has happened
                # First draw line showing position of L6
                cv2.line(self.im_resized, tuple([math.floor(self.resize_factor * self.L6), 0]),
                         tuple(
                             [math.floor(self.resize_factor * self.L6), math.floor(self.resize_factor * 511)]),
                         (0, 0, 255), 1)
                cv2.putText(self.im_resized, "L6",
                            (math.floor(self.resize_factor * self.L6), math.floor(self.resize_factor * 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), lineType=cv2.LINE_AA)
                # Now deal with reference points chosen by the user
                if len(self.pts) > 0:
                    # Sort ref point by increasing x value (= caudo-rostral position)
                    pts_arr = np.array(self.pts)
                    pts_arr_sorted = pts_arr[np.argsort(pts_arr[:, 0])]
                    for i in range(len(self.pts)):  # Draw ref points
                        cv2.circle(self.im_resized, tuple(pts_arr_sorted[i]), 3, (0, 255, 0), -1)
                    if self.closest is not None:  # Update closest point
                        cv2.circle(self.im_resized, tuple(self.closest), 10, (255, 255, 0), 1)
                    if len(self.pts) > 1:
                        # If more than one ref point, compute midpoints
                        # midpoint = center of vertebrae (between two sucessive reference points)
                        # midpnts_top and _bottom are used to draw line perpendicular to axis of vertebrae
                        self.midpnt = np.zeros((pts_arr_sorted.shape[0]-1,2)).astype(int) # coordinates of midpoints
                        midpnt_top = np.zeros((pts_arr_sorted.shape[0]-1,2)).astype(int)
                        midpnt_bot = np.zeros((pts_arr_sorted.shape[0]-1,2)).astype(int)
                        # Compute coordinates
                        for i in range(0,self.midpnt.shape[0]):
                            self.midpnt[i, :] = np.rint(np.sum(pts_arr_sorted[i:(i + 2), :], axis = 0) / 2).astype(int)
                            midlen = np.array(self.midpnt[i, :]-pts_arr_sorted[i,:])[[1,0]] * [1,-1]
                            midpnt_top[i,:] = self.midpnt[i, :] + midlen
                            midpnt_bot[i, :] = self.midpnt[i, :] - midlen
                        # Now add labels (name of vertebrae)
                        # First, find interval in which L6 position falls:
                        vname = [''] * self.midpnt.shape[0]  # Initialize list containing names to display
                        ii = int(np.digitize(math.floor(self.resize_factor * self.L6), pts_arr_sorted[:, 0]))
                        if ii > 0 and ii < pts_arr_sorted.shape[0]:
                            for n in range(0,self.midpnt.shape[0]):
                                vname[n] = self.V_ID_list[25 + ii-n-1]  # 25 = 'L6'
                            self.V_ID = vname
                        # Now calculate coordinates of label for each vertebrae
                        tcoord = np.rint(midpnt_bot + 20 * utils.Norml(midpnt_bot - self.midpnt)).astype(int)
                        for i in range(0, pts_arr_sorted.shape[0] - 1):
                            # Show midpoint
                            cv2.circle(self.im_resized, tuple(self.midpnt[i,:]), 2, (0, 255, 255), -1)
                            # Line perpendicular to caudo-rostral axis of vertebrae
                            cv2.line(self.im_resized, tuple(midpnt_top[i,:]), tuple(midpnt_bot[i,:]), (0, 255, 255), 1)
                            # Add name of vertebrae
                            cv2.putText(self.im_resized, vname[i], (tcoord[i,0], tcoord[i,1]),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), lineType=cv2.LINE_AA)
                    cv2.imshow(self.msg, self.im_resized)  # Update display
                self.something_changed = False

            if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
                cv2.destroyWindow(self.msg)
                break  # Return to main function

            time.sleep(0.01)  # Slow down while loop to reduce CPU usage

    def callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append([x, y])
            self.closest = [x, y] # Update coordinates of closest reference point
            self.closest_last = self.closest
            self.im_resized = self.im_copy.copy()
            self.something_changed = True
        if event == cv2.EVENT_MBUTTONDOWN and (len(self.pts) > 0):
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts),2),1)
            del self.pts[diff.argmin()]
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts), 2), 1)
            self.closest = self.pts[diff.argmin()]
            self.im_resized = self.im_copy.copy()
            self.closest_last = self.closest
            self.something_changed = True
        if event == cv2.EVENT_MOUSEMOVE and (len(self.pts) > 0):
            # Find reference point closest to latest mouse position (= closest reference point)
            diff = np.sum(np.power(np.tile([x, y], [len(self.pts), 1]) - np.array(self.pts),2),1)
            self.closest = self.pts[diff.argmin()]
            # Update coordinates of closest reference point
            if self.closest_last != self.closest:
                self.im_resized = self.im_copy.copy()
                self.closest_last = self.closest
                self.something_changed = True
