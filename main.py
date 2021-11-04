
############################################
# main.py                                  #
# Main function of PyMicroCT project       #
# FUNCTIONS USED TO ANALYZE MICRO-CT SCANS #
############################################

import sys, os
from analysis import run_analysis, vertebral_profiles, vertebral_angles
import utilities as utils

if __name__ == '__main__':
    # Assuming main.py is launched from appropirate folder...
    # ... e.g. 'BY926_24_Colonne_165213'
    # os.chdir('/home/ghyomm/DATA_MICROCT/SPINE/20201014_SPINE/CN723_40_Colonne_111610')
    wd = os.getcwd()
    if not os.path.exists(os.path.join(wd,'data')):
        sys.exit('No data folder in current directory!')
    list_dirs = utils.splitall(wd)
    # e.g. list_dirs =
    if not (('Colonne' in list_dirs[-1]) and ('SPINE' in list_dirs[-2])):
        sys.exit('Current folder name looks suspicious.')
    if list_dirs[-3]!='SPINE':
        sys.exit('Not in righ folder.')
    if len(sys.argv)==1 or sys.argv[1]=='rois':
        run_analysis(session = list_dirs[-2], mouse = list_dirs[-1], datapath='/mnt/data/DATA_MICROCT', struct='SPINE')
    if len(sys.argv)>1 and sys.argv[1]=='vert':
        vertebral_profiles(session = list_dirs[-2], mouse = list_dirs[-1])
    if len(sys.argv)>1 and sys.argv[1]=='sym':
        vertebral_angles(session = list_dirs[-2], mouse = list_dirs[-1])
