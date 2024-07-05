from keras.api import optimizers
from keras.src.layers import LSTM, Dense
from keras.src.utils import plot_model
from nilearn import datasets
from nilearn import image as img
from nilearn import plotting
from nilearn.input_data import NiftiMapsMasker
import numpy as np
from keras import utils, Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf

# 뇌를 smith_atlas 에 따라 70개의 구역 으로 나눈다.

smith_atlas = datasets.fetch_atlas_smith_2009()
smith_atlas_rs_networks = smith_atlas.rsn70

# Showing results
plotting.plot_prob_atlas(smith_atlas_rs_networks,
                         title='Smith atlas',
                         colorbar=True)
plotting.show()

# Import 40 subjects from a preprocessed, ready-to-go, dataset
adhd_data = datasets.fetch_adhd(n_subjects=40)

print(len(adhd_data['func']))

# Generate a mask
masker = NiftiMapsMasker(maps_img=smith_atlas_rs_networks,  # Smith atlas
                         standardize=True,  # centers and norms the time-series
                         memory='nilearn_cache',  # cache
                         verbose=0)  # do not print verbose

# Apply the mask on the data & sort it based on label

all_subjects_data = []  # actiavtion patterns for all subjects
labels = []  # 1 if ADHD, 0 if control

for func_file, confound_file, phenotypic in zip(
        adhd_data.func, adhd_data.confounds, adhd_data.phenotypic):
    time_series = masker.fit_transform(func_file, confounds=confound_file)

    all_subjects_data.append(time_series)
    labels.append(phenotypic['adhd'])

print('N control:', labels.count(0))
print('N ADHD:', labels.count(1))

# 40명의 데이터는 어쩌면 동일한 time stamp 를 가지고 있지 않을 수 있다. 그러면 데이터가 통일성 이 없어서 모델이 잘 작동 하지 않는다.

# get longest scan
max_len_image = np.max([len(i) for i in all_subjects_data])

# reshape all data
all_subjects_data_reshaped = []
for subject_data in all_subjects_data:
    # Padding
    N = max_len_image - len(subject_data)
    padded_array = np.pad(subject_data, ((0, N), (0, 0)),
                          'constant', constant_values=(0))
    subject_data = padded_array
    subject_data = np.array(subject_data)
    subject_data.reshape(subject_data.shape[0], subject_data.shape[1], 1)
    all_subjects_data_reshaped.append(subject_data)

print('data shape: ', np.array(all_subjects_data_reshaped).shape)


# Data preparation

def get_train_test(X, y, i, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.2, random_state=i)

    # Reshapes data to 3D for Hierarchical RNN.
    t_shape = np.array(all_subjects_data_reshaped).shape[1]
    RSN_shape = np.array(all_subjects_data_reshaped).shape[2]

    X_train = np.reshape(X_train, (len(X_train), t_shape, RSN_shape))
    X_test = np.reshape(X_test, (len(X_test), t_shape, RSN_shape))

    # enforce continuous labeling
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # print if verbose
    if verbose:
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

    # Converts class vectors to binary class matrices.
    y_train = utils.to_categorical(y_train, 2)
    y_test = utils.to_categorical(y_test, 2)

    return X_train, X_test, y_train, y_test


# create the model

model = tf.keras.models.Sequential()

# LSTM layers -
# Long Short-Term Memory layer - Hochreiter 1997.
t_shape = np.array(all_subjects_data_reshaped).shape[1]
RSN_shape = np.array(all_subjects_data_reshaped).shape[2]

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

