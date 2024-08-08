import pandas as pd
import Functional_Connectivity
import os
import numpy as np

# 데이터를 넣을 데이터 프레임을 하나 만든다.

jun_df = pd.DataFrame(index=None)

jun_df['FC'] = None
jun_df['Status'] = None

rbd = []
hc = []

r1 = Functional_Connectivity.RBD_FC('/Users/oj/Desktop/post_fMRI_RBD/sub-01/ses-1/func/sub-01_ses-1_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii',
                                    '/Users/oj/Desktop/post_fMRI_RBD/sub-01/ses-1/func/sub-01_ses-1_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.tsv')

r2 = Functional_Connectivity.RBD_FC('/Users/oj/Desktop/post_fMRI_RBD/sub-02/ses-1/func/sub-02_ses-1_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii',
                                    '/Users/oj/Desktop/post_fMRI_RBD/sub-02/ses-1/func/sub-02_ses-1_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.tsv')

h1 = Functional_Connectivity.HC_FC('/Users/oj/Desktop/pre_BIDS/BIDS_HC_final/sub-02/func/warsub-02_task-RESEARCHMRI_acq-AxialfMRIrest_bold.nii',
                                    '/Users/oj/Desktop/pre_BIDS/BIDS_HC_final/sub-02/func/rp_sub-02_task-RESEARCHMRI_acq-AxialfMRIrest_bold.tsv')

h2 = Functional_Connectivity.HC_FC('/Users/oj/Desktop/pre_BIDS/BIDS_HC_final/sub-03/func/warsub-03_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_rec-BRAINMRINONCONTRASTDIFFUSION_bold.nii',
                                    '/Users/oj/Desktop/pre_BIDS/BIDS_HC_final/sub-03/func/rp_sub-03_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_rec-BRAINMRINONCONTRASTDIFFUSION_bold.tsv')

rbd.append(np.array(r1.flatten()))
rbd.append(np.array(r2.flatten()))

hc.append(np.array(h1.flatten()))
hc.append(np.array(h2.flatten()))



for j in rbd:
    jun_df = pd.concat([jun_df, pd.DataFrame({'FC':[j], 'Status':[1]})], ignore_index=True)

for j in hc:
    jun_df = pd.concat([jun_df, pd.DataFrame({'FC':[j], 'Status':[0]})], ignore_index=True)

print(jun_df.head())





