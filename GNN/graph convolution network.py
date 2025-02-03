import os
import torch
import pandas as pd
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nilearn import datasets
from nilearn import plotting
from nilearn import input_data
from nilearn import image
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d
from torch_geometric.nn import EdgeConv, GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import from_networkx

import networkx as nx
## from_numpy_matrix는 더이상 지원되지 않고, 대신 from_numpy_array가 유사한 기능을 수행함.
from networkx.convert_matrix import from_numpy_array

atlas_path = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'

path = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_positive/RBD_PET_positive/sub-14/func/sub-14_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

confound_path = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_positive/RBD_PET_positive/sub-14/func/sub-14_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.tsv'

confounds = pd.read_csv(confound_path, sep='\t')

confounds.replace([np.inf, -np.inf], np.nan, inplace=True)
confounds.fillna(0, inplace=True)

shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True)

data = image.load_img(path)

time_series = shen_atlas.fit_transform(data, confounds=confounds)

print(time_series.shape)
