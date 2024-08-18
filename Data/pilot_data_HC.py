import pandas as pd
import os
import numpy as np
import glob
from Data.pilot_data_RBD import pilot_data

root_dir = '/Users/oj/Desktop/post_fMRI/post_XCP-D_HC'

reho_path_hc = glob.glob(os.path.join(root_dir, 'sub-*', 'func', '*_seg-Tian_stat-reho_bold.tsv'))
alff_path_hc = glob.glob(os.path.join(root_dir, 'sub-*', 'func', '*_seg-Tian_stat-alff_bold.tsv'))
fc_path_hc = glob.glob(os.path.join(root_dir, 'sub-*', 'func', '*_seg-Tian_stat-pearsoncorrelation_relmat.tsv'))

alff = []
reho = []
fc = []

for file_path in reho_path_hc:
    data = pd.read_csv(file_path, sep='\t')
    reho.append(np.array(data.iloc[0]))

for file_path in alff_path_hc:
    data = pd.read_csv(file_path, sep='\t')
    alff.append(np.array(data.iloc[0]))

for file_path in fc_path_hc:
    data = pd.read_csv(file_path, sep='\t')
    fc.append(data.iloc[:, 1:].values)

for k in range(len(reho)):
    pilot_data.loc[k + 3] = [fc[k], alff[k], reho[k], 0]

print(len(pilot_data['ALFF'][0]))
