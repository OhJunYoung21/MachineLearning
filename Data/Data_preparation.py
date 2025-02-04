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

atlas_path = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'

avg_pcorr_file = f'{root_dir}/GNN/avg_pcorr.csv'

fMRI_img = glob.glob(os.path.join(root_dir, 'sub-*', 'func',
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

raw_confounds = glob.glob(os.path.join(root_dir, 'sub-*', 'func',
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

raw_confounds = sorted(raw_confounds)
fMRI_img = sorted(fMRI_img)

shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True, memotry='nilearn_cache')

time_series_list = []

for index in range(len(raw_confounds)):
    fmri_img = fMRI_img[index]
    confounds = pd.read_csv(raw_confounds[index], sep='\t')

    confounds.replace([np.inf, -np.inf], np.nan, inplace=True)
    confounds.fillna(0, inplace=True)

    data = image.load_img(fmri_img)

    time_series = shen_atlas.fit_transform(data, confounds=confounds)

    time_series_list.append(time_series)
    '''
    np.savetxt(f'/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_positive/GNN/time_series/time_series_{index + 1:02d}.csv',
               time_series,
               delimiter=',')
    '''

corr_measure = ConnectivityMeasure(kind='correlation')
pcorr_measure = ConnectivityMeasure(kind='partial correlation')

corr_matrices = corr_measure.fit_transform(time_series_list)
pcorr_matrices = pcorr_measure.fit_transform(time_series_list)

print(corr_matrices[0].shape)
