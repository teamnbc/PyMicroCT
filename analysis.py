

import os, cv2, dicom, roi, time, pickle, sys
import numpy as np
import utilities as utils
from datetime import date
from scipy.interpolate import interp1d

# Drawing functions and related classes are in roi/customROI.py

'''
Steps:
1. Define paths
2. Obtain 3D array from Dicom images
3. Draw horizontal line (tilt angle will be used to rotate spine)
4. Draw mouse contour on rear view (to eliminate non-mouse objects, e.g. tube)
5. Draw contour of spine on side view (to have cleaner spine side view)s
6. Draw spine axis on top view (with spline interpolation)
7. draw limits of each vertebrae on projection obtained after step 4.
8: compute 3D coordinates of vertebrae limits
9: save coordinates of vertebrae and params
10: compute vertebrae profiles using projection plane perpendicular to spine axis

Array names (note: masks are applied to float 32 bits version...
...which are then converted to uint8):
arr3d_8/32: original 3D array (8 for uint8, 32 for float 32 bits).
arr3d_8/32_masked1: rear mask applied to arr3d_32.
arr3d_8/32_masked2: top (spline) mask applied to arr3d_8/32_masked1.
arr3d_8/32_masked3: side mask applied to arr3d_32_masked2.
'''

def run_analysis(session, mouse, datapath='/home/ghyomm/DATA_MICROCT', struct='SPINE'):
    # datapath='/home/ghyomm/DATA_MICROCT'
    # struct='SPINE'
    # session='20200202_SPINE'
    # mouse='BY908_29_Colonne_114959'
    #
    ''' Step 1: define paths '''
    analysis_path = os.path.join(datapath, struct, session, mouse, 'analysis')
    im_path = os.path.join(analysis_path, 'images')
    if not os.path.exists(im_path):
        os.makedirs(im_path)
    roi_path = os.path.join(analysis_path, 'rois')
    if not os.path.exists(roi_path):
        os.makedirs(roi_path)
    src_dir = os.path.join(datapath, struct, session, mouse, 'data')

    ''' Step 2: obtain 3D array and voxel info using all dcm files in directory '''
    arr3d_32, voxel_spacing = dicom.load_stack(src_dir)  # 3D array in float 32
    arr3d_8 = utils.imRescale2uint8(arr3d_32)  # Scale [min max] to [0 255] (i.e. convert to 8 bit)
    # Below is just for illustration purposes (saving individual dicom images)
    # for i in range(512):
    #     im = arr3d_8[i,:,:]
    #     #im = utils.imRescale2uint8(utils.imLevels(im, 10, 240))
    #     cv2.rectangle(im,(0,0),(511,511),(150,150,150),2)
    #     cv2.imwrite('/home/ghyomm/Desktop/tmp/' + "%03d" % i + '.png',im)

    ''' Step 3-5: draw horizontal line '''
    im_rear = np.amax(arr3d_8, axis=0)  # Projection (rear view)
    im_rear_rgb = cv2.merge(((im_rear,) * 3))  # Convert to RGB
    im_rear_rgb_copy = im_rear_rgb.copy()  # Copy used for drawing
    im_side = np.flip(np.transpose(np.amax(arr3d_8, axis=2)), axis=1)  # Projection (side view)
    im_side_rgb = cv2.merge(((im_side,) * 3))  # Convert to RGB
    # Then initialize class custROI1 (see customROI.py)
    roi1 = roi.CustROI1(imsrc_horiz=im_rear_rgb,
                    msg_horiz='Draw mouse\'s horizontal line (press \'q\' to escape)', fact_horiz=3,
                    imsrc_rear=im_rear_rgb_copy,
                    msg_rear='Draw mouse contour (press \'q\' to escape)', fact_rear=3,
                    imsrc_side=im_side_rgb,
                    msg_side='Draw spine contour (press \'q\' to escape)', fact_side=3)
    roi1.tilt.horizontal_line()  # Draw horizontal line underneath mouse on rear projection
    cv2.imwrite(os.path.join(im_path, 'roi1_horiz_im.png'), roi1.tilt.im_resized)
    roi1.tilt.vertical_line()
    cv2.imwrite(os.path.join(im_path, 'roi1_vert_im.png'), roi1.tilt.im_resized)
    roi1.rear.drawpolygon()  # Draw mouse's contour on rear view to define rear mask
    # cv2.imwrite(os.path.join(im_path, 'roi1_rear_im.png'), roi1.rear.im)
    # cv2.imwrite(os.path.join(im_path, 'roi1_rear_immask.png'), roi1.rear.immask)
    roi1.side.drawpolygon()  # Draw contour of spine on side view
    # cv2.imwrite(os.path.join(im_path, 'roi1_side_im.png'), roi1.side.im)
    # cv2.imwrite(os.path.join(im_path, 'roi1_side_immask.png'), roi1.side.immask)
    # print('Applying masks and preparing top view... ')
    sys.stdout.write('Applying masks and preparing top view... ')
    # Apply rear mask on original 3d array
    mask_rear = np.broadcast_to(roi1.rear.immask == 0, arr3d_32.shape)
    minval = np.ma.array(arr3d_32, mask=mask_rear).min()  # min pixel val within selected region
    arr3d_32_masked1 = np.ma.array(arr3d_32, mask=mask_rear, fill_value=minval).filled()
    arr3d_8_masked1 = utils.imRescale2uint8(arr3d_32_masked1)  # Convert to 8 bits
    # Compute and save projections (for illustration purposes)
    im_rear = utils.imRescale2uint8(utils.imLevels(utils.imRescale2uint8(np.amax(arr3d_8_masked1, axis=0)), 70, 220))
    im_side = utils.imRescale2uint8(utils.imLevels(utils.imRescale2uint8(np.amax(arr3d_8_masked1, axis=2)), 70, 220))
    im_top = utils.imRescale2uint8(utils.imLevels(utils.imRescale2uint8(np.amax(arr3d_8_masked1, axis=1)), 70, 220))
    cv2.imwrite(os.path.join(im_path, 'projection_rear.png'), im_rear)
    cv2.imwrite(os.path.join(im_path, 'projection_side.png'), np.flip(np.transpose(im_side), axis=1))
    cv2.imwrite(os.path.join(im_path, 'projection_top.png'), im_top)
    print('done.')

    ''' Step 6: draw spine axis on top view '''
    im_top_rgb = cv2.merge(((im_top,) * 3))  # Convert to RGB (to draw color lines and points)
    # Initialize CustROI2 object (see customROI.py)
    roi2 = roi.CustROI2(imsrc_top = im_top_rgb,
                        msg_top = 'Draw spine axis (press \'q\' to escape)',
                        fact_top = 3)
    roi2.top.DrawL6()  # Indicate position of L6 vertebrae
    roi2.top.DrawSpline()  # Select reference points along spine; splines are drawn automatically
    # Save images for illustration purposes
    # cv2.imwrite(os.path.join(im_path, 'roi2_top_im.png'), roi2.top.im)
    # cv2.imwrite(os.path.join(im_path, 'roi2_top_immask.png'), roi2.top.immask)
    # cv2.imwrite(os.path.join(im_path, 'roi2_top_im_annotated.png'), roi2.top.im_resized)
    # NOTE: reference points are in roi2.top.pts and still in im * resize_factor coordinates
    # Apply top mask on 3d array
    arr = np.transpose(np.flip(arr3d_32_masked1, axis=1), (1, 0, 2))  # Necessary transformation to use np.broadcast
    mask_top = np.broadcast_to(roi2.top.immask == 0, arr.shape)
    minval = np.ma.array(arr, mask=mask_top).min()  # min pixel val within selected region
    arr3d_32_masked2 = np.ma.array(arr, mask=mask_top, fill_value=minval).filled()
    arr3d_8_masked2 = utils.imRescale2uint8(arr3d_32_masked2)  # Convert to 8 bits
    # Apply side mask (just for aesthetics, so only the vertebrae are visible)
    # Below: necessary transformation to use np.broadcast
    arr = np.transpose(np.flip(np.flip(arr3d_32_masked2, axis=1), axis=0), (2, 0, 1))
    mask_side = np.broadcast_to(roi1.side.immask == 0, arr.shape)
    minval = np.ma.array(arr, mask=mask_side).min()  # min pixel val within selected region
    arr3d_32_masked3 = np.ma.array(arr, mask=mask_side, fill_value=minval).filled()
    arr3d_8_masked3 = utils.imRescale2uint8(arr3d_32_masked3)  # Convert to 8 bits
    # Compute and save projections
    im_side_through_spine = np.amax(arr3d_8_masked3, axis=0)
    im_side_through_spine = utils.imRescale2uint8(utils.imLevels(im_side_through_spine, 70, 220))
    # Shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    # Apply the sharpening kernel
    im_side_through_spine_sharpened = cv2.filter2D(im_side_through_spine, -1, kernel_sharpening)
    cv2.imwrite(os.path.join(im_path, 'im_side_through_spine.png'), im_side_through_spine_sharpened)

    ''' Step 7: draw limits of vertebrae on side projection '''
    im_side_through_spine_sharpened_rgb = cv2.merge(((im_side_through_spine_sharpened,) * 3))
    roi3 = roi.CustROI3(imsrc_side=im_side_through_spine_sharpened_rgb,
                    msg_side='Draw limits of vertebrae (press \'q\' to escape)', fact_side=3,
                    L6_position=512-roi2.top.L6_pos)
    roi3.side.DrawVertebrae()
    cv2.imwrite(os.path.join(im_path, 'im_side_through_spine_annotated.png'), roi3.side.im_resized)
    # NOTE: reference points are in roi3.side.pts and still in im * resize_factor coordinates

    ''' Step 8: compute 3D coordinates of vertebrae limits '''
    # Define coordinates of reference points drawn on top projection
    pts_top_arr = np.array(roi2.top.pts)
    pts_top_arr_sorted = np.flip(pts_top_arr[np.argsort(pts_top_arr[:, 1])])
    pts_top_arr_sorted = -pts_top_arr_sorted
    pts_top_arr_sorted[:, 0] = pts_top_arr_sorted[:, 0] - pts_top_arr_sorted[0, 0]
    # in pts_top_arr_sorted, x is ascending caudo-rostral coordinate, y is ascending ML coordinate (from right to left)
    # Define coordinates of reference points drawn on side projection (vertebrae limits)
    pts_side_arr = np.array(roi3.side.pts)
    pts_side_arr_sorted = pts_side_arr[np.argsort(pts_side_arr[:, 0])]
    pts_side_arr_sorted[:, 1] = -pts_side_arr_sorted[:, 1] + 512 * roi3.side.resize_factor
    # in pts_side_arr_sorted, x is ascending caudo-rostral coordinate, y is ascending ventro-dorsal coordinate
    # Same for midpoints (center of vertebrae)
    midpts = np.array(roi3.side.midpnt)
    midpts[:, 1] = -midpts[:, 1] + 512 * roi3.side.resize_factor
    # Compute spline joining pts_top_arr_sorted
    x, y = np.array(pts_top_arr_sorted)[:, 1], np.array(pts_top_arr_sorted)[:, 0]
    f = interp1d(y, x, kind='cubic')  # Compute spline function
    # ynew = np.linspace(np.min(y), np.max(y), num=500, endpoint=True)
    # spline_top = np.vstack((np.array(f(ynew)), ynew)).T  # Coordinates of spline points
    # spline_top[:,0] = -spline_top[:,0]
    # spline_top[:,1] = spline_top[:,1] - spline_top[0,1]
    # in spline.top, x is ascending left-to-right coordinate, y is ascending caudo-rostral coordinate
    ynew = pts_side_arr_sorted[:, 0]
    vlimits = np.vstack((np.array(f(ynew)), ynew)).T  # Coordinates of spline points
    vlimits[:, 0] = vlimits[:, 0] + 512 * roi3.side.resize_factor
    # vlimits: coordinates of vertebrae limits, same coordinate system as spline.top...
    # ...(x is ascending left-to-right coordinate, y is ascending caudo-rostral coordinate)
    ynew = midpts[:, 0]
    center_v = np.vstack((np.array(f(ynew)), ynew)).T  # Coordinates of spline points
    center_v[:, 0] = center_v[:, 0] + 512 * roi3.side.resize_factor

    ''' Step 9: save coordinates of vertebrae and params '''
    N_V_annotated = len(roi3.side.V_ID)  # Number of vertebrae annotated
    voxel_size = int(round(np.mean(np.array(voxel_spacing)) * 1000))
    # roi3.side.V_ID.append("-")
    # center_v = np.vstack((center_v, np.array([np.nan, np.nan])))
    # Save coordinates of vertebrae limits into file together with vertebrae ID
    with open(os.path.join(analysis_path, 'vertebrae_coordinates.txt'), 'w') as outfile:
        outfile.write("ID\tx_cen\ty_cen\tz_cen\tx_lim\ty_lim\tz_lim\n")
        for i in range(pts_side_arr_sorted.shape[0]-1):
            x_cen = (center_v[i, 1]/roi3.side.resize_factor)*(voxel_size/1000)
            y_cen = (center_v[i, 0]/roi3.side.resize_factor)*(voxel_size/1000)
            z_cen = (midpts[i, 1]/roi3.side.resize_factor)*(voxel_size/1000)
            x_lim = (pts_side_arr_sorted[i, 0]/roi3.side.resize_factor)*(voxel_size/1000)
            y_lim = (vlimits[i, 0]/roi3.side.resize_factor)*(voxel_size/1000)
            z_lim = (pts_side_arr_sorted[i, 1]/roi3.side.resize_factor)*(voxel_size/1000)
            outfile.write(roi3.side.V_ID[i] + "\t")  # ID
            outfile.write("%f" % x_cen + "\t")  # x_cen
            outfile.write("%f" % y_cen + "\t")  # y_cen
            outfile.write("%f" % z_cen + "\t")  # z_cen
            outfile.write("%f" % x_lim + "\t")  # x_lim
            outfile.write("%f" % y_lim + "\t")  # y_lim
            outfile.write("%f" % z_lim + "\n")  # z_lim
        i = pts_side_arr_sorted.shape[0]-1
        x_lim = (pts_side_arr_sorted[i, 0] / roi3.side.resize_factor) * (voxel_size / 1000)
        y_lim = (vlimits[i, 0] / roi3.side.resize_factor) * (voxel_size / 1000)
        z_lim = (pts_side_arr_sorted[i, 1] / roi3.side.resize_factor) * (voxel_size / 1000)
        outfile.write("-\tnan\tnan\tnan\t")
        outfile.write("%f" % x_lim + "\t")  # x_lim
        outfile.write("%f" % y_lim + "\t")  # y_lim
        outfile.write("%f" % z_lim + "\n")  # z_lim
    # Save relevant params
    today = date.today()
    with open(os.path.join(analysis_path, 'params.txt'), 'w') as outfile:
        outfile.write(os.path.join(datapath, struct, session, mouse) + "\n")
        outfile.write("Date = " + today.strftime("%Y-%m-%d") + "\n")
        outfile.write("Horizontal tilt angle = " + str(roi1.tilt.hangle) + " deg\n")
        outfile.write("Vertical tilt angle = " + str(roi1.tilt.vangle) + " deg\n")
        outfile.write("Voxel size = " + str(tuple([int(round(1000 * x)) for x in voxel_spacing])) + " um\n")
        outfile.write("roi2.top.resize_factor = " + str(roi2.top.resize_factor) + "\n")
        outfile.write("roi3.side.resize_factor = " + str(roi3.side.resize_factor) + "\n")
        outfile.write("Number of vertebrae annotated = " + str(N_V_annotated) + "\n")

    ''' Step 10: save ROI objects in file '''
    for i in range(1,4):
        roi_name = 'roi' + str(i)
        with open(os.path.join(roi_path, roi_name + '.pickle'), 'wb') as f:
            pickle.dump(locals()[roi_name], f)

def vertebral_profiles(session, mouse, datapath='/home/ghyomm/DATA_MICROCT', struct='SPINE'):
    '''
    Ongoing work: this function will take on the analysis from step 10 on.
    Need to regenerate/load the following variables:
    (better to save all roi objects using pickle)
    - arr3d_8_masked1
    - number of vertebrae annotated
    '''

    # session='20200202_SPINE'
    # mouse='BY908_29_Colonne_114959'

    ''' Step 1: define paths '''
    analysis_path = os.path.join(datapath, struct, session, mouse, 'analysis')
    im_path = os.path.join(analysis_path, 'images')
    roi_path = os.path.join(analysis_path, 'rois')
    src_dir = os.path.join(datapath, struct, session, mouse, 'data')

    '''Step 2: load roi objects '''
    roi1 = pickle.load(open(os.path.join(analysis_path, 'rois', 'roi1.pickle'),'rb'))
    roi2 = pickle.load(open(os.path.join(analysis_path, 'rois', 'roi2.pickle'),'rb'))
    roi3 = pickle.load(open(os.path.join(analysis_path, 'rois', 'roi3.pickle'),'rb'))

    '''Step 3: regenerate arrays from data (see run_analysis function)'''
    arr3d_32, voxel_spacing = dicom.load_stack(src_dir)
    arr3d_8 = utils.imRescale2uint8(arr3d_32)
    mask_rear = np.broadcast_to(roi1.rear.immask == 0, arr3d_32.shape)
    minval = np.ma.array(arr3d_32, mask=mask_rear).min()  # min pixel val within selected region
    arr3d_32_masked1 = np.ma.array(arr3d_32, mask=mask_rear, fill_value=minval).filled()
    arr3d_8_masked1 = utils.imRescale2uint8(arr3d_32_masked1)  # Convert to 8 bits

    ''' Step 4: compute 3D coordinates of vertebrae limits (same as in run_analysis function)'''
    # Define coordinates of reference points drawn on top projection
    pts_top_arr = np.array(roi2.top.pts)
    pts_top_arr_sorted = np.flip(pts_top_arr[np.argsort(pts_top_arr[:, 1])])
    pts_top_arr_sorted = -pts_top_arr_sorted
    pts_top_arr_sorted[:, 0] = pts_top_arr_sorted[:, 0] - pts_top_arr_sorted[0, 0]
    # in pts_top_arr_sorted, x is ascending caudo-rostral coordinate, y is ascending ML coordinate (from right to left)
    # Define coordinates of reference points drawn on side projection (vertebrae limits)
    pts_side_arr = np.array(roi3.side.pts)
    pts_side_arr_sorted = pts_side_arr[np.argsort(pts_side_arr[:, 0])]
    pts_side_arr_sorted[:, 1] = -pts_side_arr_sorted[:, 1] + 512 * roi3.side.resize_factor
    # in pts_side_arr_sorted, x is ascending caudo-rostral coordinate, y is ascending ventro-dorsal coordinate
    # Same for midpoints (center of vertebrae)
    midpts = np.array(roi3.side.midpnt)
    midpts[:, 1] = -midpts[:, 1] + 512 * roi3.side.resize_factor
    # Compute spline joining pts_top_arr_sorted
    x, y = np.array(pts_top_arr_sorted)[:, 1], np.array(pts_top_arr_sorted)[:, 0]
    f = interp1d(y, x, kind='cubic')  # Compute spline function
    # ynew = np.linspace(np.min(y), np.max(y), num=500, endpoint=True)
    # spline_top = np.vstack((np.array(f(ynew)), ynew)).T  # Coordinates of spline points
    # spline_top[:,0] = -spline_top[:,0]
    # spline_top[:,1] = spline_top[:,1] - spline_top[0,1]
    # in spline.top, x is ascending left-to-right coordinate, y is ascending caudo-rostral coordinate
    ynew = pts_side_arr_sorted[:, 0]
    vlimits = np.vstack((np.array(f(ynew)), ynew)).T  # Coordinates of spline points
    vlimits[:, 0] = vlimits[:, 0] + 512 * roi3.side.resize_factor
    # vlimits: coordinates of vertebrae limits, same coordinate system as spline.top...
    # ...(x is ascending left-to-right coordinate, y is ascending caudo-rostral coordinate)
    ynew = midpts[:, 0]
    center_v = np.vstack((np.array(f(ynew)), ynew)).T  # Coordinates of spline points
    center_v[:, 0] = center_v[:, 0] + 512 * roi3.side.resize_factor

    ''' Step 5: compute vertebrae profiles using projection plane perpendicular to spine axis'''
    # We use the midpoints calculated in run_analysis
    # u, v, w: reference frame
    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])
    w = np.array([0, 0, 1])
    arr = np.flip(np.flip(np.swapaxes(arr3d_8_masked1, 2, 1), 0), 2)
    v_analysis_path = os.path.join(im_path, 'vertebrae')  # Where images will be saved
    if not os.path.exists(v_analysis_path):
        os.makedirs(v_analysis_path)
    # print('Computing oblique projections through vertebrae... ', end='')
    sys.stdout.write('Computing oblique projections through vertebrae... ')
    N_V_annotated = len(roi3.side.V_ID)  # Number of vertebrae annotated
    for k in range(N_V_annotated):
        TV1 = pts_side_arr_sorted[k + 1] - pts_side_arr_sorted[k]
        TV2 = vlimits[k + 1] - vlimits[k]
        vec = utils.norml(np.array([TV1[0],TV2[0],TV1[1]]))
        up = utils.norml(utils.vprod(v, vec))
        vp = utils.norml(utils.vprod(vec, up))
        ori = np.array([midpts[k, 0], 3*512-center_v[k, 0], midpts[k, 1]]) / 3
        v_im1 = np.zeros((71, 71))
        v_im2 = np.zeros((71, 71))
        v_im3 = np.zeros((71, 71))
        v_im4 = np.zeros((71, 71))
        v_im5 = np.zeros((71, 71))
        for i in np.linspace(-35,35,num=71):
            for j in np.linspace(-35,35,num=71):
                coord = np.round(ori - (2*vec) + i * up + j * vp).astype('int16')
                if not np.any((coord >= 512) | (coord <= 0)):
                    v_im1[int(i + 35), int(j + 35)] = arr[coord[0], coord[1], coord[2]]
                else:
                    v_im1[int(i + 35), int(j + 35)] = 0
                coord = np.round(ori - vec + i*up + j*vp).astype('int16')
                if not np.any((coord >= 512) | (coord <= 0)):
                    v_im2[int(i + 35), int(j + 35)] = arr[coord[0], coord[1], coord[2]]
                else:
                    v_im2[int(i + 35), int(j + 35)] = 0
                coord = np.round(ori + i*up + j*vp).astype('int16')
                if not np.any((coord >= 512) | (coord <= 0)):
                    v_im3[int(i + 35), int(j + 35)] = arr[coord[0], coord[1], coord[2]]
                else:
                    v_im3[int(i + 35), int(j + 35)] = 0
                coord = np.round(ori + vec + i*up + j*vp).astype('int16')
                if not np.any((coord >= 512) | (coord <= 0)):
                    v_im4[int(i + 35), int(j + 35)] = arr[coord[0], coord[1], coord[2]]
                else:
                    v_im4[int(i + 35), int(j + 35)] = 0
                coord = np.round(ori + (2*vec) + i * up + j * vp).astype('int16')
                if not np.any((coord >= 512) | (coord <= 0)):
                    v_im5[int(i + 35), int(j + 35)] = arr[coord[0], coord[1], coord[2]]
                else:
                    v_im5[int(i + 35), int(j + 35)] = 0
        v_im = np.amax(np.array([v_im1, v_im2, v_im3, v_im4, v_im5]), axis=0)
        v_im_resized, resize_factor = utils.customResize(v_im, 3)
        ret, v_im_bin = cv2.threshold(v_im_resized, 100, 255, cv2.THRESH_BINARY)
        # v_im_sharpened = cv2.filter2D(v_im, -1, kernel_sharpening)
        # v_im_resized, resize_factor = utils.customResize(v_im, 3)
        # v_im_resized = utils.imRescale2uint8(utils.imLevels(v_im_resized, 70, 220))
        v_im_bin_rgb = cv2.merge(((v_im_bin,) * 3))
        cv2.line(v_im_bin_rgb, (107 - 10, 107), (107 + 10, 107), (0, 0, 255), 2)
        cv2.line(v_im_bin_rgb, (107, 107 - 10), (107, 107 + 10), (0, 0, 255), 2)
        cv2.putText(v_im_bin_rgb, roi3.side.V_ID[k], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.rectangle(v_im_bin_rgb, (0, 0), (int(71*resize_factor-1), int(71*resize_factor-1)), (130, 130, 130), 1)
        cv2.imwrite(os.path.join(v_analysis_path, "%02d" % k + '_' + roi3.side.V_ID[k] + '.png'), v_im_bin_rgb)
    print('done.')
