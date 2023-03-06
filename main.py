
'''
Main PyMicroCT script.
Assuming main.py is launched from appropriate folder containing subfolder ./data/
Good idea to have an alias for executing main.py, e.g. in ~/.bash_aliases:
alias pmct='python3 /path_to/PyMicroCT/main.py'
Example:
start_folder/       -> where main.py is executed.
    ├── analysis    -> folder created by PyMicroCT.
    └── data        -> original data folder.
        └── a single .dcm stack or a set of individual .dmc images.
'''

import sys, os
from analysis import run_analysis, vertebral_profiles, vertebral_angles
import utilities as utils

if __name__ == '__main__':
    # Debug:
    # os.chdir('/mnt/data/DATA_MICROCT/SPINE/20201014_SPINE/CN723_40_Colonne_111610')
    # os.chdir('/mnt/data/DATA_SSPO/SPINE/example_session_SPINE/example_mouse_Colonne')
    # os.chdir('/mnt/data/DATA_Micro-CT_CR_CHUSJ')

    wd = os.getcwd()
    if not os.path.exists(os.path.join(wd,'data')):
        sys.exit('No \"data\" subfolder in current directory!')
    list_dirs = utils.splitall(wd)
    if not (('Colonne' in list_dirs[-1]) and ('SPINE' in list_dirs[-2])):
        sys.exit('Current folder name looks suspicious.')
    if list_dirs[-3]!='SPINE':
        sys.exit('Not in right folder.')
    if len(sys.argv)==1 or sys.argv[1]=='rois':
        run_analysis(session = list_dirs[-2], mouse = list_dirs[-1])
    if len(sys.argv)>1 and sys.argv[1]=='vert':
        vertebral_profiles(session = list_dirs[-2], mouse = list_dirs[-1])
    if len(sys.argv)>1 and sys.argv[1]=='sym':
        vertebral_angles(session = list_dirs[-2], mouse = list_dirs[-1])
