from nilearn import plotting
from nilearn import datasets
from nilearn import image
from nilearn.input_data import NiftiMapsMasker
import numpy as np
import keras
from keras import utils, Sequential
from sklearn.model_selection import train_test_split
from torch.nn import LSTM

# Smith's re-fMRI component atlas 를 적용

smith_atlas = datasets.fetch_atlas_smith_2009()
smith_atlas_rs_networks = smith_atlas.rsn70

adhd_data = datasets.fetch_adhd(n_subjects=40)

# Generate a mask
masker = NiftiMapsMasker(maps_img=smith_atlas_rs_networks,  # Smith atlas
                         standardize=True,  # centers and norms the time-series
                         memory='nilearn_cache',  # cache
                         verbose=0)  # do not print verbose

all_subjects_data = []
labels = []

print("done")

for func_file, confound_file, phenotypic in zip(
        adhd_data.func, adhd_data.confounds, adhd_data.phenotypic):
    time_series = masker.fit_transform(func_file, confounds=confound_file)

    all_subjects_data.append(time_series)
    labels.append(phenotypic['adhd'])

# labels 에는 실제 데이터가 들어있다.

print('N control:', labels.count(0))
print('N ADHD:', labels.count(1))

max_len_image = np.max([len(i) for i in all_subjects_data])

# Data might not hold homogenous scan, some of them might not be in same time length with others

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


def get_train_test(X, y, i, verbose=False):
    """
    split to train and test and reshape data

    X data
    y labels
    i random state
    """

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.1, random_state=i)

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

model = Sequential()

# LSTM layers -
# Long Short-Term Memory layer - Hochreiter 1997.
t_shape = np.array(all_subjects_data_reshaped).shape[1]
RSN_shape = np.array(all_subjects_data_reshaped).shape[2]

model.add(LSTM(units=70,  # dimensionality of the output space
               dropout=0.4,  # Fraction of the units to drop (inputs)
               recurrent_dropout=0.15,  # Fraction of the units to drop (recurent state)
               return_sequences=True,  # return the last state in addition to the output
               input_shape=(t_shape, RSN_shape)))

model.add(LSTM(units=60,
               dropout=0.4,
               recurrent_dropout=0.15,
               return_sequences=True))

model.add(LSTM(units=50,
               dropout=0.4,
               recurrent_dropout=0.15,
               return_sequences=True))

model.add(LSTM(units=40,
               dropout=0.4,
               recurrent_dropout=0.15,
               return_sequences=False))

model.add(Dense(units=2,
                activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.001),
              metrics=['binary_accuracy'])

#print(model.summary())
plot_model(model, show_shapes=True, show_layer_names=True)
