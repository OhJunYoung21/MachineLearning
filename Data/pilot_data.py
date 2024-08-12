import pandas as pd
import os
import numpy as np

pilot_data = pd.DataFrame(index=None)

pilot_data['FC'] = None
pilot_data['ALFF'] = None
pilot_data['REHO'] = None
pilot_data['STATUS'] = None


reho = []
alff = []
fc = []

pilot_reho = pd.read_csv(
    '/Users/oj/Desktop/post_fMRI/xcp_d_HC/sub-03/func/sub-03_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_seg-Tian_stat-reho_bold.tsv',
    sep='\t')

print(pilot_reho)


reho.append(np.array(pilot_reho.iloc[0]))


for j in reho:
    pilot_data = pd.concat([pilot_data, pd.DataFrame({'REHO':[j], 'Status':[0]})], ignore_index=True)

print(pilot_data.head())
