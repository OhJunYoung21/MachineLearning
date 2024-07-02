from nilearn import plotting
from nilearn import datasets
from nilearn import image
from nilearn.input_data import NiftiMapsMasker

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

print('N control:' ,labels.count(0))
print('N ADHD:' ,labels.count(1))
