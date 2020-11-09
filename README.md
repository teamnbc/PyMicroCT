# PyMicroCT

![alt text](https://github.com/ghyomm/PyMicroCT/blob/master/pics/Repo%20card.png)

PyMicroCT is a set of Python classes and functions designed to analyze microCT scans of the mouse spine. It uses opencv and numpy to
draw and apply user-defined masks, and provides annotation functions to reconstruct the 3D axis of the spine, label individual vertebrae and compute vertebrae cross-sections.

## Pre-Prequisites

    pip3 install opencv-python
    pip3 install pydicom

## Steps of the analysis
1. A stack of microCT images (Dicom files) is loaded using Pydicom and converted into a 3D numpy array.
2. An estimation of the tilt of the animal is performed.
2. The user draws bounding boxes on rear and side projections of the 3D array.
4. He/she draws reference points along the spine's midline on a top projection, from which a spline interpolation is performed. A narrow longitudinal band centered around the spline is used as a mask to only keep a thin stripe of midline pixels in the 3D array.
5. He/she then selects intervertebral spaces on a side projection of these pixels. Because the user is first asked to indicate the position of the L6 vertebra, vertebral bodies are automatically labeled using the appropriate name (C1 to C7, T1 to T13, L1 to L6 and S1 to S4; see Vertebrae_ID.png).
6. The 3D coordinates of these intervertebral reference points are calculated and a cross-section image is calculated for each vertebra through a plane normal to the spine's 3D axis.
