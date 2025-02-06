import os
import pandas as pd
import numpy as np

from nilearn import datasets
from nilearn import plotting
from nilearn import input_data
from nilearn import image
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
import glob
import re

root_dir = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_positive'

fMRI_img_positive = glob.glob(os.path.join(root_dir, 'sub-*', 'func',
                                           'sub-*_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-DIFFUSIONMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-fMRIRESTSENSE_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))

raw_confounds_positive = glob.glob(os.path.join(root_dir, 'sub-*', 'func',
                                                'sub-*_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.tsv')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_desc-confounds_timeseries.tsv')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-DIFFUSIONMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_desc-confounds_timeseries.tsv')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-fMRIRESTSENSE_desc-confounds_timeseries.tsv')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-AxialfMRIrest_desc-confounds_timeseries.tsv')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest_desc-confounds_timeseries.tsv'))

positive_path = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_positive'

positive_folders = [name for name in os.listdir(positive_path)
                    if os.path.isdir(os.path.join(positive_path, name)) and name.startswith("sub-")]

'''
nagative_path = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_negative'

negative_folders = [name for name in os.listdir(positive_path)
                    if os.path.isdir(os.path.join(positive_path, name)) and name.startswith("sub-")]
'''


class Data:
    def __init__(self, func=None, confounds=None, phenotypic=None):
        self.func = func
        self.phenotypic = phenotypic
        self.confounds = confounds


fMRI_img_positive = sorted(fMRI_img_positive)

func = []

for j in fMRI_img_positive:
    func_img = image.load_img(j)
    func.append(func_img)

confounds_list = []

for k in range(len(raw_confounds_positive)):
    confounds = pd.read_csv(raw_confounds_positive[k], sep='\t')

    confounds.replace([np.inf, -np.inf], np.nan, inplace=True)
    confounds.fillna(0, inplace=True)

    confounds_list.append(confounds)

subject_numbers = [f"{i:02d}" for i in range(1, len(positive_folders) + 1)]

labels = [1 for i in range(len(positive_folders))]

phenotypic = [[subject_numbers[i], labels[i]] for i in range(len(positive_folders))]

data = Data(func=func, confounds=confounds_list, phenotypic=phenotypic)

print(data.func)
print(data.phenotypic)
print(data.confounds)
