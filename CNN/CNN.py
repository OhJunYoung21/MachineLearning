import tensorflow as tf
from tensorflow.keras import layers, models
from Data.pilot_data_HC import pilot_data
from tensorflow.keras.applications.vgg16 import VGG16

# weight, include_top 파라미터 설정
model = VGG16(weights='imagenet', include_top=True)
model.summary()

