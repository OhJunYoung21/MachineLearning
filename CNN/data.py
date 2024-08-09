import os
import pandas as pd

df = pd.read_csv(
    '/Users/oj/Desktop/post_fMRI/xcp_d/sub-03/func/sub-03_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_seg-Tian_stat-pearsoncorrelation_relmat.tsv',
    sep='\t')

print(df.head())
