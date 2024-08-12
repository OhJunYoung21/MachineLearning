import os
import pandas as pd

df_FC = pd.read_csv(
    '/Users/oj/Desktop/post_fMRI/xcp_d_HC/sub-03/func/sub-03_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_seg-Tian_stat-pearsoncorrelation_relmat.tsv',
    sep='\t')


df_reho = pd.read_csv(
    '/Users/oj/Desktop/post_fMRI/xcp_d_HC/sub-03/func/sub-03_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_seg-Tian_stat-reho_bold.tsv',
    sep='\t')



print(df_reho.head())
