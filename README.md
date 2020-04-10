# PyMicroCT

This is a set of functions to analyze microCT scans of the mouse spine.

## Data organization

Data folder structure :
datapath (in my case /home/ghyomm/DATA_MicroCT)
 |--SPINE
 |   |--20190724_SPINE (imaging session)
 |   |   |--BY926_24_Colonne_165213 (folder containing Dicom files for one stack)
 |   |   |   |--BY926_24_Colonne_165213_0001.dcm (dcm = Dicom file)
 |   |   |   |--...
 |   |   |   |--BY926_24_Colonne_165213_0512.dcm (there are 512 images per stack)
 |   |   |--BY927_23_Colonne_164250 (another mouse...)
 |   |   |--...
 |   |--20190725_SPINE (another imaging session...)
 |   |--...
 |-INNEREAR
     |--same organization as for SPINE
