import pandas as pd
import os
import numpy as np
import glob

pilot_data = pd.DataFrame(index=None)

pilot_data['FC'] = None
pilot_data['ALFF'] = None
pilot_data['REHO'] = None
pilot_data['STATUS'] = None

root_dir = '/Users/oj/Desktop/post_fMRI/post_XCP-D_RBD'

alff = []
reho = []
fc = []

## 디렉토리내의 sub들의 reho를 전부 추출해서 reho 리스트안에 넣는다.

reho_path_rbd = glob.glob(os.path.join(root_dir, 'sub-*', 'func', '*_seg-Tian_stat-reho_bold.tsv'))
alff_path_rbd = glob.glob(os.path.join(root_dir, 'sub-*', 'func', '*_seg-Tian_stat-alff_bold.tsv'))
fc_path_rbd = glob.glob(os.path.join(root_dir, 'sub-*', 'func', '*_seg-Tian_stat-pearsoncorrelation_relmat.tsv'))

for file_path in reho_path_rbd:
    data = pd.read_csv(file_path, sep='\t')
    reho.append(np.array(data.iloc[0]))

for file_path in alff_path_rbd:
    data = pd.read_csv(file_path, sep='\t')
    alff.append(np.array(data.iloc[0]))

for file_path in fc_path_rbd:
    data = pd.read_csv(file_path, sep='\t')
    fc.append(data.iloc[:, 1:].values)

for k in range(len(reho)):
    pilot_data.loc[k] = [fc[k], alff[k], reho[k], 1]

print(pilot_data.head())

## 디렉토리내의 sub들의 alff를 전부 추출해서 reho 리스트안에 넣는다.

## 디렉토리내의 sub들의 fc를 전부 추출해서 reho 리스트안에 넣는다.
