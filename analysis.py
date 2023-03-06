
'''
Main PyMicroCT functions (run_analysis, vertebral_profiles and vertebral_angles) are here.
'''

# Python modules:
import os, cv2, pydicom, time, pickle, sys, glob
import numpy as np
from datetime import date
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

# Custom modules:
import dicom  # Custom module to bundle several .dcm files into one 3D numpy array.
import utilities as utils
import roi
import symmetry as sym
from cython_code import cyutils


def run_analysis(session, mouse, datapath='/mnt/data/DATA_SSPO', struct='SPINE'):

    # Change default string for arg datapath depending on dataset:
    # datapath='/mnt/data/DATA_SSPO' for SSPO mice.
    # datapath='/mnt/data/DATA_MICROCT' for POC5 mice.
    # datapath='/mnt/data/DATA_Micro-CT_CR_CHUSJ' for CHUSJ mice.

    # For debug:
    # datapath='/mnt/data/DATA_SSPO'
    # struct='SPINE'
    # session='SSPO_cohort-1_SPINE'
    # mouse='Mouse_53-2_Colonne'

    ''' Step 1: define paths '''

    subj_path = os.path.join(datapath, struct, session, mouse)  # Animal directory.
    data_path = os.path.join(subj_path, 'data')  # Location of DICOM (.dcm) file(s).
    if not os.path.exists(data_path):
        sys.exit('No data folder in ' + subj_path + '.')
    analysis_path = os.path.join(subj_path, 'analysis')  # Where pmct will store its files.
    im_path = os.path.join(analysis_path, 'images')  # Where pmct will store images.
    if not os.path.exists(im_path):
        os.makedirs(im_path)
    roi_path = os.path.join(analysis_path, 'rois')  # Where pmct will store roi information.
    if not os.path.exists(roi_path):
        os.makedirs(roi_path)

    ''' Step 2: obtain 3D array and voxel info from .dcm file(s) '''

    # Debug for CHUSJ mice:
    # data_path = '/mnt/data/DATA_Micro-CT_CR_CHUSJ/SPINE/DICOM/2M/2M-25J-1'

    dcm_files = glob.glob(data_path + '/*.dcm')  # Looking for .dcm files in data folder.
    ndcm = len(dcm_files)  # Number of .dcm files.

    if ndcm == 0:  # Case .dcm file(s) missing.
        sys.exit('No .dcm file in ' + data_path + '.')

    if ndcm > 1:  # Case several .dcm files.
        # Assuming data folder contains .dcm files with incremented names, belonging to same stack, e.g.:
        # mouseXXX_sessionYY_0001.dcm
        # mouseXXX_sessionYY_0002.dcm
        # ...
        # mouseXXX_sessionYY_0512.dcm
        # Using package dicom to read file.
        # Tested with .dcm files obtained with a Micro-CT Quantum FX (Perkin Elmer).

        # For debug:
        # data_path = '/mnt/data/DATA_MICROCT/SPINE/20190724_SPINE/BY927_23_Colonne_164250/data'

        arr3d_float32, voxel_spacing = dicom.load_stack(data_path)  # Usually a 3D array of floats.

    if ndcm == 1:  # Case one dcm file.
        # Assuming data folder has one .dcm file containing a stack of images.
        # Using package pydicom to read file.
        # Assuming file is built using the official DICOM Standard.
        # Useful information here: https://pydicom.github.io/pydicom/stable/tutorials/dataset_basics.html
        # Tested with a .dcm stack exported via Amide from data obtained using an Albira system (Bruker).

        # For debug:
        # src_dir = '/mnt/data/DATA_SSPO/SPINE/example_session_SPINE/example_mouse_Colonne/data/export_windows.dcm'

        data = pydicom.dcmread(dcm_files[0])
        # data.NumberOfSlices
        voxel_spacing = (float(data.SliceThickness),float(data.PixelSpacing[0]),float(data.PixelSpacing[1]))  # voxel size in mm.
        arr = data.pixel_array
        arr3d_float32 = np.float32(arr)

    arr3d_8 = utils.imRescale2uint8(arr3d_float32)  # Scale [min max] to [0 255] (i.e. convert to 8 bit).

    # The stack has (should have) the following dimensions:
    # Dimension 0 (X): rostro-caudal direction.
    # -> slices along dimension 0 are coronal slices (index 0 = most rostral slice).
    # Dimension 1 (Y): dorso-ventral direction.
    # -> slices along dimension 1 are transverse slices (index 0 = most dorsal pixel).
    # Dimension 2 (Z): medio-lateral axis.
    # -> slices along dimension 2 are (para)sagittal slices (index 0 = right side of animal).

    # Main assumption: when looking at transverse slices (top view), left of image is left of animal.
    # This assumptions sets the following right-handed reference frame:
    #
    #                           Y axis
    #                          (dorsal)
    #                             0
    #                             |
    # (tail) arr.shape[0]-1 ------------ 0 (nose) X axis
    #                             |
    #                       arr.shape[1]-1
    #                         (ventral)
    #
    # Third axis (Z) along medio-lateral axis pointing toward observer (from LEFT to RIGHT).

    # Saving individual dicom images:
    # for i in range(512):
    #     im = arr3d_8[:,:,i]
    #     im = utils.imRescale2uint8(utils.imLevels(im, 10, 240))
    #     cv2.imwrite('/mnt/data/DATA_SSPO/' + "%03d" % i + '.png',im)

    ''' Step 3: initialize class CustROI1 for working on rear and side views '''

    # Calculate rear view:
    # max projection along caudo-rostral axis, nose pointing away from observer;
    # the left of animal is on the left side of the resulting image.
    # Note the flip to obtain image in right orientation.
    im_rear = np.amax(arr3d_8, axis=0)  # Projection along caudo-rostral axis (rear view).
    im_rear = utils.imRescale2uint8(utils.imLevels(im_rear,utils.imBackground(im_rear),255))
    cv2.imwrite(os.path.join(im_path, 'ORIGINAL_rear_view.png'), im_rear)  # For illustration purposes.
    im_rear_rgb = cv2.merge(((im_rear,) * 3))  # Convert to RGB.
    # Dimensions of resulting 2D array:
    # Dimension 0 (rows): dorso-ventral direction (0 = most dorsal pixels).
    # Dimension 1 (columns): medio-lateral axis (0 = left side of animal, left side of image).
    # i.e.:
    # (left of animal, (0,0) ------------ (0,ncols)     (right of animal,
    # dorsal)          |                  |             dorsal)
    #                  |                  |
    # (left of animal, |                  |             (right of animal,
    # ventral)         (nrows,0) -------- (nrows,ncols) ventral)

    # Calculate side view:
    # Note the transpose and flip steps to obtain image in proper orientation.
    im_side = np.flip(np.transpose(np.amax(arr3d_8, axis=2)), axis=1)  # Projection (side view)
    im_side = utils.imRescale2uint8(utils.imLevels(im_side,utils.imBackground(im_side),255))
    cv2.imwrite(os.path.join(im_path, 'ORIGINAL_side_view.png'), im_side)  # For illustration purposes.
    im_side_rgb = cv2.merge(((im_side,) * 3))  # Convert to RGB
    # Dimensions of resulting 2D array:
    # Dimension 0 (rows): dorso-ventral direction (0 = most dorsal pixels).
    # Dimension 1 (columns): caudo-rostral axis (0 = tail of animal, left of image).
    # i.e.:
    # (tail of animal, (0,0) ------------ (0,ncols)     (nose of animal,
    # dorsal)          |                  |             dorsal)
    #                  |                  |
    # (tail of animal, |                  |             (nose of animal,
    # ventral)         (nrows,0) -------- (nrows,ncols) ventral)

    # Calculate top view (not used in CustROI1 but for illustration purposes):
    # Note the flip to bring the right of the animal on the right of the image.
    im_top = np.amax(arr3d_8, axis=1)
    im_top = utils.imRescale2uint8(utils.imLevels(im_top,utils.imBackground(im_top),255))
    cv2.imwrite(os.path.join(im_path, 'ORIGINAL_top_view.png'), im_top)
    # Dimensions of resulting 2D array:
    # Dimension 0 (rows): rostro-caudal direction (0 = toward nose).
    # Dimension 1 (columns): medio-lateral axis (0 = left of animal, left of image).
    # i.e.:
    # (left of animal, (0,0) -------- (0,ncols)     (right of animal,
    # rostral)         |              |             rostral)
    #                  |              |
    #                  |              |
    #                  |              |
    #                  |              |
    # (left of animal, |              |             (right of animal,
    # caudal)          (nrows,0) ---- (nrows,ncols) ventral)

    # Build class CustROI1:
    im_rear_rgb_copy = im_rear_rgb.copy()  # Copy used for drawing.
    roi1 = roi.CustROI1(imsrc_horiz=im_rear_rgb,
                    msg_horiz='Draw mouse\'s horizontal line (press \'q\' to escape)', fact_horiz=3,
                    imsrc_rear=im_rear_rgb_copy,
                    msg_rear='Draw mouse contour (press \'q\' to escape)', fact_rear=3,
                    imsrc_side=im_side_rgb,
                    msg_side='Draw spine contour (press \'q\' to escape)', fact_side=3)

    ''' Step 4: annotate rear and side views and apply rear mask '''

    # Draw horizontal line underneath mouse on rear view:
    roi1.tilt.horizontal_line()
    cv2.imwrite(os.path.join(im_path, 'ROI1_horiz_line.png'), roi1.tilt.im_resized)

    # Draw vertical line (= animal symetry line) on rear view:
    roi1.tilt.vertical_line()
    cv2.imwrite(os.path.join(im_path, 'ROI1_vert_line.png'), roi1.tilt.im_resized)

    # Draw mouse's contour on rear view to define rear mask.
    roi1.rear.drawpolygon()
    cv2.imwrite(os.path.join(im_path, 'ROI1_rear_roi.png'), roi1.rear.im)
    cv2.imwrite(os.path.join(im_path, 'ROI1_rear_mask.png'), roi1.rear.immask)

    # Draw contour of spine on side view to define side mask.
    roi1.side.drawpolygon()
    cv2.imwrite(os.path.join(im_path, 'ROI1_side_roi.png'), roi1.side.im)
    cv2.imwrite(os.path.join(im_path, 'ROI1_side_mask.png'), roi1.side.immask)

    # Apply rear mask on original 3d array:
    mask_rear = np.broadcast_to(roi1.rear.immask == 0, arr3d_float32.shape)  # Compute rear mask.
    arr3d_float32_masked1 = np.ma.array(arr3d_float32, mask=mask_rear, fill_value=arr3d_float32.min()).filled()
    arr3d_8_masked1 = utils.imRescale2uint8(arr3d_float32_masked1)  # Convert to uint8.

    # Save masked projections (for illustration purposes):
    im_rear_masked = np.amax(arr3d_8_masked1, axis=0)
    im_rear_masked = utils.imRescale2uint8(utils.imLevels(im_rear_masked,utils.imBackground(im_rear_masked),255))
    cv2.imwrite(os.path.join(im_path, 'MASKED_ROI1_rear_view.png'), im_rear_masked)
    im_side_masked = np.flip(np.transpose(np.amax(arr3d_8_masked1, axis=2)), axis=1)
    im_side_masked = utils.imRescale2uint8(utils.imLevels(im_side_masked,utils.imBackground(im_side_masked),255))
    cv2.imwrite(os.path.join(im_path, 'MASKED_ROI1_side_view.png'), im_side_masked)
    im_top_masked = np.amax(arr3d_8_masked1, axis=1)
    im_top_masked = utils.imRescale2uint8(utils.imLevels(im_top_masked,utils.imBackground(im_top_masked),255))
    cv2.imwrite(os.path.join(im_path, 'MASKED_ROI1_top_view.png'), im_top_masked)

    ''' Step 5: initialize class CustROI2 for working on top view '''

    im_top_masked_rgb = cv2.merge(((im_top_masked,) * 3))  # Convert to RGB (to draw color lines and points)
    roi2 = roi.CustROI2(imsrc_top = im_top_masked_rgb,
                        msg_top = 'Draw spine axis (press \'q\' to escape)',
                        fact_top = 3, width = (1,3))

    ''' Step 6: annotate top view and sequentially apply top and side masks '''

    # Draw position of L6 vertebrae:
    roi2.top.DrawL6()

    # Select reference points along spine; splines are drawn automatically:
    roi2.top.DrawSpline()  # NOTE: reference points are in roi2.top.pts in resized coordinates.

    # Save images for illustration purposes:
    cv2.imwrite(os.path.join(im_path, 'ROI2_top_spline.png'), roi2.top.im)
    cv2.imwrite(os.path.join(im_path, 'ROI2_top_spline_mask.png'), roi2.top.immask)

    # Apply top mask on 3d array:
    # Note: the array is flipped and transposed before apply mask...
    # ...because of the mismatch between numpy and opencv (rows,columns) indexing.
    arr = np.transpose(np.flip(arr3d_float32_masked1, axis=1), (1, 0, 2))  # Necessary transformation to use np.broadcast
    mask_top = np.broadcast_to(roi2.top.immask == 0, arr.shape)
    arr3d_float32_masked2 = np.ma.array(arr, mask=mask_top, fill_value=arr3d_float32_masked1.min()).filled()
    arr3d_8_masked2 = utils.imRescale2uint8(arr3d_float32_masked2)  # Convert to uint8.

    # Save top view obtained after applying top mask (for illustration purposes):
    im_top_masked2 = np.amax(arr3d_8_masked2, axis=0)
    im_top_masked2_vals = np.sort(np.unique(im_top_masked2.flatten()))
    minval = np.delete(im_top_masked2_vals,np.where(im_top_masked2_vals == 0)[0][0])[0]
    maxval = max(im_top_masked2_vals)
    im_top_masked2 = utils.imRescale2uint8(utils.imLevels(im_top_masked2,minval,maxval))
    cv2.imwrite(os.path.join(im_path, 'ROI2_top_view_masked.png'), im_top_masked2)

    # Apply side mask, so only the vertebrae are visible:
    # Note: again, flipping and transposing is necessary before applying mask.
    arr = np.transpose(np.flip(np.flip(arr3d_float32_masked2, axis=1), axis=0), (2, 0, 1))
    mask_side = np.broadcast_to(roi1.side.immask == 0, arr.shape)
    arr3d_float32_masked3 = np.ma.array(arr, mask=mask_side, fill_value=arr3d_float32_masked2.min()).filled()
    arr3d_8_masked3 = utils.imRescale2uint8(arr3d_float32_masked3)  # Convert to uint8.

    # Compute and save side projection with both top and side masks applied:
    im_side_through_spine = np.amax(arr3d_8_masked3, axis=0)
    im_side_through_spine = utils.imRescale2uint8(utils.imLevels(im_side_through_spine,utils.imBackground(im_side_through_spine),255))
    cv2.imwrite(os.path.join(im_path, 'ROI2_side_through_spine.png'), im_side_through_spine)
    # Apply shapening kernel (which must equal to one eventually):
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    im_side_through_spine_sharpened = cv2.filter2D(im_side_through_spine, -1, kernel_sharpening)
    cv2.imwrite(os.path.join(im_path, 'ROI2_side_through_spine_sharpened.png'), im_side_through_spine_sharpened)

    ''' Step 7: initialize CustROI3 for working on side view through spine '''

    im_side_through_spine_sharpened_rgb = cv2.merge(((im_side_through_spine_sharpened,) * 3))
    roi3 = roi.CustROI3(imsrc_side=im_side_through_spine_sharpened_rgb,
                    msg_side='Draw limits of vertebrae (press \'q\' to escape)', fact_side=3,
                    L6_position=im_side_through_spine_sharpened.shape[1]-1-roi2.top.L6_pos)

    ''' Step 8: draw limits of vertebrae on side projection '''

    roi3.side.DrawVertebrae()
    cv2.imwrite(os.path.join(im_path, 'ROI3_side_annotated.png'), roi3.side.im_resized)
    # NOTE: reference points are in roi3.side.pts in resized coordinates.

    ''' Step 9: compute 3D coordinates of vertebrae limits '''

    # Define coordinates of reference points drawn on top projection:
    pts_top_arr = np.array(roi2.top.pts)
    pts_top_arr_sorted = np.flip(pts_top_arr[np.argsort(pts_top_arr[:, 1])])
    pts_top_arr_sorted = -pts_top_arr_sorted
    pts_top_arr_sorted[:, 0] = pts_top_arr_sorted[:, 0] - pts_top_arr_sorted[0, 0]
    # in pts_top_arr_sorted, x is ascending caudo-rostral coordinate, y is ascending ML coordinate (from RIGHT to LEFT).
    # plt.plot(*pts_top_arr_sorted.T,'o')

    # Define coordinates of reference points drawn on side projection (vertebrae limits):
    pts_side_arr = np.array(roi3.side.pts)
    pts_side_arr_sorted = pts_side_arr[np.argsort(pts_side_arr[:, 0])]
    pts_side_arr_sorted[:, 1] = -pts_side_arr_sorted[:, 1] + arr3d_8.shape[1] * roi3.side.resize_factor
    # in pts_side_arr_sorted, x is ascending caudo-rostral coordinate, y is ascending ventro-dorsal coordinate
    # plt.plot(*pts_side_arr_sorted.T,'o')

    # Same for midpoints (center of vertebrae):
    midpts = np.array(roi3.side.midpnt)
    midpts_sorted = midpts[np.argsort(midpts[:, 0])]
    midpts_sorted[:, 1] = -midpts_sorted[:, 1] + arr3d_8.shape[1] * roi3.side.resize_factor
    # in midpts_sorted, x is ascending caudo-rostral coordinate, y is ascending ventro-dorsal coordinate
    # plt.plot(*midpts_sorted.T,'o')

    # Compute spline joining pts_top_arr_sorted
    x, y = pts_top_arr_sorted[:, 1], pts_top_arr_sorted[:, 0]  # x: ascending ML coordinate (from RIGHT to LEFT); y: ascending caudo-rostral coordinate.
    f = interp1d(y, x, kind='cubic')  # Compute spline function.

    # Control of result of spline interpolation:
    # ynew = np.linspace(np.min(y), np.max(y), num=500, endpoint=True)
    # spline_top = np.vstack((np.array(f(ynew)), ynew)).T  # Coordinates of spline points
    # plt.plot(*spline_top.T,'o')
    # In spline_top: x is ML coordinate from RIGHT to LEFT, y is ascending caudo-rostral coordinate,

    # Interpolation of vertebra limits (i.e. reference points) coordinate along media-lateral axis.
    ynew = pts_side_arr_sorted[:, 0]  # ascending caudo-rostral coordinate.
    vlimits = np.vstack((np.array(f(ynew)), ynew)).T  # Coordinates of spline points
    vlimits[:, 0] = vlimits[:, 0] + arr3d_8.shape[1] * roi3.side.resize_factor
    # in vlimits: x is ascending RIGHT to LEFT coordinate, y is ascending caudo-rostral coordinate.
    # plt.plot(*vlimits.T,'o')

    # Interpolation of vertebrae midpoints coordinate along media-lateral axis.
    ynew = midpts_sorted[:, 0]
    center_v = np.vstack((np.array(f(ynew)), ynew)).T  # Coordinates of spline points
    center_v[:, 0] = center_v[:, 0] + arr3d_8.shape[1] * roi3.side.resize_factor
    # in center_v: x is ascending RIGHT to LEFT coordinate, y is ascending caudo-rostral coordinate.
    # plt.plot(*center_v.T,'o')

    ''' Step 10: save coordinates of vertebrae and params '''

    N_V_annotated = len(roi3.side.V_ID)  # Number of vertebrae annotated.
    voxel_size = voxel_spacing[0]  # Assuming voxel size is the same in all dimensions.

    # Save coordinates of vertebrae limits into file together with vertebrae ID.
    # The final reference frame is a bit different from the original one:
    # it is a right handed reference frame with:
    # - increasing x values along the caudo-rostral direction (positive x toward nose).
    # - increasing y values along the media-lateral axis with positive y toward left side.
    # - increasing z values along the ventro-dorsal axis with positive z toward dorsal side.
    with open(os.path.join(analysis_path, 'vertebrae_coordinates.txt'), 'w') as outfile:
        outfile.write("ID\tx_cen\ty_cen\tz_cen\tx_lim\ty_lim\tz_lim\n")
        for i in range(pts_side_arr_sorted.shape[0]-1):
            x_cen = (center_v[i, 1]/roi3.side.resize_factor)*voxel_size  # Same as midpts_sorted[:, 0].
            y_cen = (center_v[i, 0]/roi3.side.resize_factor)*voxel_size  # Interpolated with spline.
            z_cen = (midpts_sorted[i, 1]/roi3.side.resize_factor)*voxel_size  # Ascending ventro-dorsal coordinate.
            x_lim = (pts_side_arr_sorted[i, 0]/roi3.side.resize_factor)*voxel_size
            y_lim = (vlimits[i, 0]/roi3.side.resize_factor)*voxel_size  # Interpolated with spline.
            z_lim = (pts_side_arr_sorted[i, 1]/roi3.side.resize_factor)*voxel_size
            outfile.write(roi3.side.V_ID[i] + "\t")  # ID
            outfile.write("%f" % x_cen + "\t")  # x_cen
            outfile.write("%f" % y_cen + "\t")  # y_cen
            outfile.write("%f" % z_cen + "\t")  # z_cen
            outfile.write("%f" % x_lim + "\t")  # x_lim
            outfile.write("%f" % y_lim + "\t")  # y_lim
            outfile.write("%f" % z_lim + "\n")  # z_lim
        # Deal with last (most rostral) vertebrae.
        i = pts_side_arr_sorted.shape[0]-1
        x_lim = (pts_side_arr_sorted[i, 0] / roi3.side.resize_factor)*voxel_size
        y_lim = (vlimits[i, 0] / roi3.side.resize_factor)*voxel_size
        z_lim = (pts_side_arr_sorted[i, 1] / roi3.side.resize_factor)*voxel_size
        outfile.write(roi3.side.V_ID[i] + "\tnan\tnan\tnan\t")
        outfile.write("%f" % x_lim + "\t")  # x_lim
        outfile.write("%f" % y_lim + "\t")  # y_lim
        outfile.write("%f" % z_lim + "\n")  # z_lim

    # Save relevant params in separate txt file.
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

    ''' Step 11: save ROI objects in file '''

    for i in range(1,4):
        roi_name = 'roi' + str(i)
        with open(os.path.join(roi_path, roi_name + '.pickle'), 'wb') as f:
            pickle.dump(locals()[roi_name], f)


def vertebral_profiles(session, mouse, datapath='/mnt/data/DATA_SSPO', struct='SPINE'):
    # Debug:
    # datapath='/mnt/data/DATA_SSPO'
    # struct='SPINE'
    # session='example_session_SPINE'
    # mouse='example_mouse_Colonne'

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
    dcm_files = glob.glob(src_dir + '/*.dcm')
    ndcm = len(dcm_files)
    if ndcm == 0:
        sys.exit('No .dcm file in ' + src_dir + '.')
    if ndcm > 1:
        arr3d_float32, voxel_spacing = dicom.load_stack(src_dir)  # 3D array in float 32
    if ndcm == 1:  # One stack of dcm images.
        dcm_data = pydicom.dcmread(dcm_files[0])
        voxel_spacing=(float(dcm_data.SliceThickness),float(dcm_data.PixelSpacing[0]),float(dcm_data.PixelSpacing[1]))  # voxel size in mm.
        arr = dcm_data.pixel_array
        arr3d_float32 = np.float32(arr)
    arr3d_8 = utils.imRescale2uint8(arr3d_float32)  # Scale [min max] to [0 255] (i.e. convert to 8 bit)
    mask_rear = np.broadcast_to(roi1.rear.immask == 0, arr3d_float32.shape)
    minval = np.ma.array(arr3d_float32, mask=mask_rear).min()  # min pixel val within selected region
    arr3d_float32_masked1 = np.ma.array(arr3d_float32, mask=mask_rear, fill_value=minval).filled()
    arr3d_8_masked1 = utils.imRescale2uint8(arr3d_float32_masked1)  # Convert to 8 bits

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
    pts_side_arr_sorted[:, 1] = -pts_side_arr_sorted[:, 1] + arr3d_8.shape[1] * roi3.side.resize_factor
    # in pts_side_arr_sorted, x is ascending caudo-rostral coordinate, y is ascending ventro-dorsal coordinate
    # Same for midpoints (center of vertebrae)
    midpts = np.array(roi3.side.midpnt)
    midpts[:, 1] = -midpts[:, 1] + arr3d_8.shape[1] * roi3.side.resize_factor
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
    vlimits[:, 0] = vlimits[:, 0] + arr3d_8.shape[1] * roi3.side.resize_factor
    # vlimits: coordinates of vertebrae limits, same coordinate system as spline.top...
    # ...(x is ascending left-to-right coordinate, y is ascending caudo-rostral coordinate)
    ynew = midpts[:, 0]
    center_v = np.vstack((np.array(f(ynew)), ynew)).T  # Coordinates of spline points
    center_v[:, 0] = center_v[:, 0] + arr3d_8.shape[1] * roi3.side.resize_factor

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
    if not os.path.exists(os.path.join(v_analysis_path, 'raw')):
        os.makedirs(os.path.join(v_analysis_path, 'raw'))
    if not os.path.exists(os.path.join(v_analysis_path, 'labeled')):
        os.makedirs(os.path.join(v_analysis_path, 'labeled'))
    # print('Computing oblique projections through vertebrae... ', end='')
    sys.stdout.write('Computing oblique projections through vertebrae... ')
    N_V_annotated = len(roi3.side.V_ID)  # Number of vertebrae annotated
    for k in range(N_V_annotated):
        TV1 = pts_side_arr_sorted[k + 1] - pts_side_arr_sorted[k]
        TV2 = vlimits[k + 1] - vlimits[k]
        vec = utils.norml(np.array([TV1[0],TV2[0],TV1[1]]))
        up = utils.norml(utils.vprod(v, vec))
        vp = utils.norml(utils.vprod(vec, up))
        ori = np.array([midpts[k, 0], 3*arr3d_8.shape[1]-center_v[k, 0], midpts[k, 1]]) / 3
        v_im1 = np.zeros((71, 71))
        v_im2 = np.zeros((71, 71))
        v_im3 = np.zeros((71, 71))
        v_im4 = np.zeros((71, 71))
        v_im5 = np.zeros((71, 71))
        v_im6 = np.zeros((71, 71))
        v_im7 = np.zeros((71, 71))
        for i in np.linspace(-35,35,num=71):
            for j in np.linspace(-35,35,num=71):
                #
                coord = np.round(ori - (3*vec) + i*up + j*vp).astype('int16')
                if not np.any((coord >= arr3d_8.shape[1]) | (coord <= 0)):
                    v_im1[int(i + 35), int(j + 35)] = arr[coord[0], coord[1], coord[2]]
                else:
                    v_im1[int(i + 35), int(j + 35)] = 0
                #
                coord = np.round(ori - (2*vec) + i*up + j*vp).astype('int16')
                if not np.any((coord >= arr3d_8.shape[1]) | (coord <= 0)):
                    v_im2[int(i + 35), int(j + 35)] = arr[coord[0], coord[1], coord[2]]
                else:
                    v_im2[int(i + 35), int(j + 35)] = 0
                #
                coord = np.round(ori - vec + i*up + j*vp).astype('int16')
                if not np.any((coord >= arr3d_8.shape[1]) | (coord <= 0)):
                    v_im3[int(i + 35), int(j + 35)] = arr[coord[0], coord[1], coord[2]]
                else:
                    v_im3[int(i + 35), int(j + 35)] = 0
                #
                coord = np.round(ori + i*up + j*vp).astype('int16')
                if not np.any((coord >= arr3d_8.shape[1]) | (coord <= 0)):
                    v_im4[int(i + 35), int(j + 35)] = arr[coord[0], coord[1], coord[2]]
                else:
                    v_im4[int(i + 35), int(j + 35)] = 0
                #
                coord = np.round(ori + vec + i*up + j*vp).astype('int16')
                if not np.any((coord >= arr3d_8.shape[1]) | (coord <= 0)):
                    v_im5[int(i + 35), int(j + 35)] = arr[coord[0], coord[1], coord[2]]
                else:
                    v_im5[int(i + 35), int(j + 35)] = 0
                #
                coord = np.round(ori + (2*vec) + i * up + j * vp).astype('int16')
                if not np.any((coord >= arr3d_8.shape[1]) | (coord <= 0)):
                    v_im6[int(i + 35), int(j + 35)] = arr[coord[0], coord[1], coord[2]]
                else:
                    v_im6[int(i + 35), int(j + 35)] = 0
                #
                coord = np.round(ori - (3*vec) + i*up + j*vp).astype('int16')
                if not np.any((coord >= arr3d_8.shape[1]) | (coord <= 0)):
                    v_im7[int(i + 35), int(j + 35)] = arr[coord[0], coord[1], coord[2]]
                else:
                    v_im7[int(i + 35), int(j + 35)] = 0
                #
        v_im = np.amax(np.array([v_im1, v_im2, v_im3, v_im4, v_im5]), axis=0)
        v_im_resized, resize_factor = utils.customResize(v_im, 3)
        v_im_resized = utils.imRescale2uint8(v_im_resized)
        ret, v_im_bin = cv2.threshold(v_im_resized, 130, 255, cv2.THRESH_BINARY)
        # v_im_sharpened = cv2.filter2D(v_im, -1, kernel_sharpening)
        # v_im_resized, resize_factor = utils.customResize(v_im, 3)
        # v_im_resized = utils.imRescale2uint8(utils.imLevels(v_im_resized, 70, 220))
        # Save binary image (greyscale)
        cv2.imwrite(os.path.join(v_analysis_path,'raw', "%02d" % k + '_' + roi3.side.V_ID[k] + '.png'), v_im_resized)
        # Then save labeled image (rgb)
        v_im_bin_rgb = cv2.merge(((v_im_bin,) * 3))
        v_im_resized_rgb = cv2.merge(((v_im_resized,) * 3))
        cv2.line(v_im_resized_rgb, (107 - 10, 107), (107 + 10, 107), (0, 0, 255), 2)
        cv2.line(v_im_resized_rgb, (107, 107 - 10), (107, 107 + 10), (0, 0, 255), 2)
        cv2.putText(v_im_resized_rgb, roi3.side.V_ID[k], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.rectangle(v_im_resized_rgb, (0, 0), (int(71*resize_factor-1), int(71*resize_factor-1)), (130, 130, 130), 1)
        cv2.imwrite(os.path.join(v_analysis_path,'labeled', "%02d" % k + '_' + roi3.side.V_ID[k] + '.png'), v_im_resized_rgb)
    print('done.')

def vertebral_angles(session, mouse, datapath='/home/ghyomm/DATA_MICROCT', struct='SPINE'):

    ''' Step 1: define paths '''
    analysis_path = os.path.join(datapath, struct, session, mouse, 'analysis')
    raw_im_path = os.path.join(analysis_path, 'images','vertebrae','raw')
    sym_im_path = os.path.join(analysis_path, 'images','vertebrae','sym')
    if not os.path.exists(sym_im_path):
        os.makedirs(sym_im_path)

    '''Step 2: loop through vertebrae images and compute symmetry'''
    res = []
    for imfile in glob.glob(raw_im_path + '/*.png'):
        im = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)
        refpt = np.array([100,100],dtype=np.uint16)
        angle_range = np.array([-30,30],dtype=np.int16)
        hoffset_range = np.array([-20,21],dtype=np.int16)
        im_out = cyutils.compute_sym_axis(im,refpt,angle_range,hoffset_range)
        max_coord = np.where(im_out == np.amax(im_out))
        if len(max_coord[0]>1) or len(len(max_coord[1]>1)):
            best_angle = int(max_coord[0][0] + angle_range[0])
            best_offset = int(max_coord[1][0] + hoffset_range[0])
        else:
            best_angle = int(max_coord[0] + angle_range[0])
            best_offset = int(max_coord[1] + hoffset_range[0])
        imsym = sym.compute_angle_and_offset(im,best_angle,best_offset)
        im_filename = os.path.split(imfile)[-1]
        index = im_filename.split('_')[0]
        ID = im_filename.split('_')[1].split('.')[0]
        res.append({'index':int(index),'ID':ID,'angle':best_angle})
        cv2.imwrite(os.path.join(sym_im_path,os.path.split(imfile)[-1]), imsym)

    df = pd.DataFrame(res).sort_values(by='index')
    df.reset_index(inplace=True,drop=True)
    del df['index']
    df.to_csv(os.path.join(analysis_path,'vertebrae_angles.txt'),sep='\t', index=True, header=True)
