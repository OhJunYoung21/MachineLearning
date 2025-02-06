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

from Data import dataset

root_dir = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_positive'

atlas_path = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'

avg_pcorr_file = f'{root_dir}/GNN/avg_pcorr.csv'
corr_matrices_dir = f'{root_dir}/GNN/corr_matrices'
pcorr_matrices_dir = f'{root_dir}/GNN/pcorr_matrices'
labels_file = f'{root_dir}/GNN/labels/labels.csv'

shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True, memotry='nilearn_cache')

time_series = [0] * len(dataset.func)
labels = [0] * len(dataset.phenotypic)

if __name__ == "__main__":
    def create_matrices():
        for index in range(0, len(dataset.func)):
            ts = shen_atlas.fit_transform(dataset.func[index], confounds=dataset.confounds[index])

            time_series[index] = ts

            labels[index] = dataset.phenotypic[index][1]

            np.savetxt(
                f'/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_positive/GNN/time_series/time_series_{index + 1:02d}.csv',
                ts,
                delimiter=',')

        corr_measure = ConnectivityMeasure(kind='correlation')
        pcorr_measure = ConnectivityMeasure(kind='partial correlation')

        corr_matrices = corr_measure.fit_transform(time_series)
        pcorr_matrices = pcorr_measure.fit_transform(time_series)

        # Get average partial correlation matrix across time series and save
        avg_pcorr_matrix = np.mean(pcorr_matrices, axis=0)
        np.savetxt(avg_pcorr_file, avg_pcorr_matrix, delimiter=',')

        # Save correlation and partial correlation matrices
        for i in range(0, len(corr_matrices)):
            ### corr_matrices_dir가 비었을 경우에만 파일들을 저장한다. 매번 실행때마다 파일들이 생성되는 수고를 덜기 위함이다.

            np.savetxt(f'{corr_matrices_dir}/corr_{i + 1:02d}.csv', corr_matrices[i], delimiter=',')

            ### pcorr_matrices_dir가 비었을 경우에만 파일들을 저장한다. 매번 실행때마다 파일들이 생성되는 수고를 덜기 위함이다.

            np.savetxt(f'{pcorr_matrices_dir}/pcorr_{i + 1:02d}.csv', pcorr_matrices[i], delimiter=',')

        return


    def create_labels():
        label_nums = [1 if label == "positive" else 0 for label in labels]

        np.savetxt(labels_file, np.asarray(label_nums).astype(int), delimiter=',')
