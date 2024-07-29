from nilearn import datasets
import os
from nilearn.image import load_img
from nilearn import plotting
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from sklearn.preprocessing import LabelEncoder

#yeo atlas를 dataset에서 가져온다.

atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas["maps"]
# Loading atlas data stored in 'labels'
labels = atlas["labels"]

masker = NiftiMapsMasker(
    maps_img=atlas_filename,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    memory="nilearn_cache",
    verbose=5,
)


def RBD_FC(image_route, confound_route):
    img = load_img(
        image_route)

    pre_confounds_RBD = pd.read_csv(
        confound_route,
        sep='\t')

    confounds = pre_confounds_RBD.apply(pd.to_numeric, errors='coerce')

    confounds = confounds.fillna(0)

    time_series_RBD = masker.fit_transform(img, confounds=confounds)

    correlation_measure = ConnectivityMeasure(
        kind="correlation",
        standardize="zscore_sample",
    )
    correlation_matrix = correlation_measure.fit_transform([time_series_RBD])[0]

    np.fill_diagonal(correlation_matrix, 0)

    return correlation_matrix


def HC_FC(image_route, confound_route):

    img = load_img(
        image_route)

    pre_confounds_RBD = pd.read_csv(
        confound_route,
        sep='\t')

    confounds = pre_confounds_RBD.apply(pd.to_numeric, errors='coerce')

    confounds = confounds.fillna(0)

    time_series_HC = masker.fit_transform(img, confounds=confounds)

    correlation_measure = ConnectivityMeasure(
        kind="correlation",
        standardize="zscore_sample",
    )
    correlation_matrix = correlation_measure.fit_transform([time_series_HC])[0]

    np.fill_diagonal(correlation_matrix, 0)

    return correlation_matrix

