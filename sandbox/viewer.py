
import numpy as np
from pathlib import Path
import pydicom as dicom
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from typing import Tuple

# # Also can use this code to play with OpenCV
# i = 259
# im_top = cv2.merge(((np.amax(arr3d_8, axis=1),) * 3))
# im_top_copy = im_top.copy()
# while True:
#     im = np.transpose(arr3d_8[::-1, :, i], (1, 0))
#     im_rgb = cv2.merge(((im,) * 3))  # Convert to RGB
#     cv2.putText(im_rgb, str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)
#     im_top = im_top_copy.copy()
#     cv2.line(im_top,(i,0),(i,511),(0, 255, 0),1)
#     cv2.imshow('blu', im_top)
#     cv2.imshow('bla', im_rgb)
#     k = cv2.waitKeyEx(33)
#     print(k)
#     if k == 65361:  # Left arrow
#         i -= 1
#     if k == 65363:  # Right arrow
#         i += 1
#     if k == 113 or k == 27:  # Escape or 'q'
#         cv2.destroyAllWindows()
#         break
#     time.sleep(0.01)


def read_stack(data_path):
    """read dicom images from a given folder and put them in a 3D array"""
    print('Reading dicom files and calculating 3d array... ')
    data_path = Path(data_path)
    files = data_path.glob('*.dcm')
    # Read slices as a list before sorting
    dcm_slices = [dicom.read_file(fname.as_posix()) for fname in files]
    # Extract position for each slice to sort and calculate slice spacing
    dcm_slices = [(dcm, thru_plane_position(dcm)) for dcm in dcm_slices]
    dcm_slices = sorted(dcm_slices, key=itemgetter(1))
    spacings = np.diff([dcm_slice[1] for dcm_slice in dcm_slices])
    slice_spacing = np.mean(spacings)
    # All slices will have the same in-plane shape
    shape = (int(dcm_slices[0][0].Columns), int(dcm_slices[0][0].Rows))
    nslices = len(dcm_slices)
    # Final 3D array will be N_Slices x Columns x Rows
    shape = (nslices, *shape)
    img = np.zeros(shape, dtype=np.uint8)
    for idx, (dcm, _) in enumerate(dcm_slices):
        # Rescale and shift in order to get accurate pixel values
        slope = float(dcm.RescaleSlope)
        intercept = float(dcm.RescaleIntercept)
        r_frame = dcm.pixel_array.astype('float32')*slope + intercept
        r_frame -= r_frame.min()
        r_frame /= r_frame.max()
        r_frame *= 255
        img[idx, ...] = r_frame.astype(np.uint8)
    # Calculate size of a voxel in mm
    pixel_spacing = tuple(float(spac) for spac in dcm_slices[0][0].PixelSpacing)
    voxel_spacing = (slice_spacing, *pixel_spacing)
    print('done.')
    return img, voxel_spacing


def thru_plane_position(dcm):
    """Get spatial coordinate of image origin whose axis is perpendicular to image plane."""
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    position = tuple((float(p) for p in dcm.ImagePositionPatient))
    rowvec, colvec = orientation[:3], orientation[3:]
    normal_vector = np.cross(rowvec, colvec)
    slice_pos = np.dot(position, normal_vector)
    return slice_pos


class CTViewer(object):
    def __init__(self, img, voxel_spacing):
        self._img, self.voxel_spacing = img, voxel_spacing
        plt.ion()
        self.fig = plt.figure()
        self._gs = GridSpec(2, 2)
        self.ax_top = self.fig.add_subplot(self._gs[1, 0])
        self.ax_side = self.fig.add_subplot(self._gs[0, 0], sharex=self.ax_top)
        self.ax_rear = self.fig.add_subplot(self._gs[1, 1], sharey=self.ax_top)
        self.axs = [self.ax_top, self.ax_side, self.ax_rear]
        shape: Tuple[int] = self.img.shape
        self._c_ix = shape[1] // 2
        self.img_side = self.ax_side.imshow(np.rot90(self.img[:, :, self.c_ix]), cmap=cm.Greys_r)
        self.img_top = self.ax_top.imshow(self.img[:, self.c_ix, :], cmap=cm.Greys_r)
        self.img_rear = self.ax_rear.imshow(self.img[self.c_ix, :, :], cmap=cm.Greys_r)
        self.imgs = [self.img_side, self.img_top, self.img_rear]
        self.c_ix = self._c_ix
        [img.set_clim((0, 255)) for img in self.imgs]
        self.ax_top.set_xlim(0, shape[2])
        self.ax_top.set_ylim(shape[0], 0)
        self.ax_side.set_ylim(0, shape[1])
        self.ax_rear.set_xlim(0, shape[2])
        self._cid_press = self.fig.canvas.mpl_connect('key_press_event',
                                                      self.on_key)

    @property
    def c_ix(self):
        return self._c_ix

    @c_ix.setter
    def c_ix(self, value):
        if value > min(self.img.shape) or value < 0:
            return
        self._c_ix = value
        self.update_img()

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, value):
        self._img = value
        self.update_img()

    @property
    def side(self):
        return np.rot90(self.img[:, :, self.c_ix]).copy()

    @property
    def top(self):
        return self.img[:, self.c_ix, :].copy()

    @property
    def rear(self):
        return self.img[self.c_ix, :, :].copy()

    def update_img(self):
        """
        Update the displayed pictures
        """
        self.img_side.set_data(np.rot90(self.img[:, :, self.c_ix]))
        self.img_top.set_data(self.img[:, self.c_ix, :])
        self.img_rear.set_data(self.img[self.c_ix, :, :])
        pos = ','.join([f'{self.c_ix * vsp:.2f} mm' for vsp in self.voxel_spacing])
        self.ax_rear.set_title(f'Slices #{self.c_ix} - {pos}')
        if self.img.dtype == np.uint8:
            [img.set_clim(0, 255) for img in self.imgs]
        else:
            [img.set_clim(0, 1) for img in self.imgs]

    def close(self):
        self.fig.canvas.mpl_disconnect(self._cid_press)
        plt.close(self.fig)

    def on_key(self, event):
        """
        React to keys pressed

        Parameters
        ----------
        event: matplotlib keypress event
        """
        if event.key == 'right':
            self.c_ix += 1
        elif event.key == 'left':
            self.c_ix -= 1
        elif event.key == 'q':
            self.close()


if __name__ == '__main__':
    DATA_PATH = '/home/ghyomm/Desktop/POC5/MicroCT/DATA/SPINE/20190723_SPINE/S1820_09_Colonne_151539/data/'
    img, vp = read_stack(DATA_PATH)
    v = CTViewer(img, vp)
