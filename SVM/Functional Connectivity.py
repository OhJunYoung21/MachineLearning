from nilearn import datasets
import os
from nilearn.image import load_img
from nilearn import plotting
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
import pandas as pd
from scipy.stats import pearsonr

#yeo atlas를 dataset에서 가져온다.

atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas["maps"]
# Loading atlas data stored in 'labels'
labels = atlas["labels"]

# 4D image uploaded

img_RBD = load_img('/Users/oj/Desktop/pre_BIDS/BIDS_RBD/sub-03/func/sub-03_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_bold.nii')

pre_confounds = pd.read_csv('/Users/oj/Desktop/pre_BIDS/BIDS_RBD/sub-03/func/rp_sub-03_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_bold.tsv',sep='\t')

confounds = pre_confounds.apply(pd.to_numeric,errors='coerce')

confounds = confounds.fillna(0)


masker = NiftiMapsMasker(
    maps_img=atlas_filename,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    memory="nilearn_cache",
    verbose=5,
)

time_series = masker.fit_transform(img_RBD,confounds = confounds)

# Pearson 상관계수를 사용해서 각 region간의 상관계수를 계산함.

correlation_measure = ConnectivityMeasure(
    kind="correlation",
    standardize="zscore_sample",
)
correlation_matrix_RBD = correlation_measure.fit_transform([time_series])[0]

print(correlation_matrix_RBD.shape)


