import tensorflow as tf
from tensorflow.keras import layers, models
from Data.pilot_data import pilot_data
import keras

import pandas as pd
import numpy as np

model = models.Sequential()

# 1. 첫 번째 합성곱 레이어
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# 2. 두 번째 합성곱 레이어
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 3. 세 번째 합성곱 레이어
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 4. 네 번째 합성곱 레이어
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


model.add(keras.layers.Dropout(0.25)())

# Flatten 레이어로 1D 벡터로 변환
model.add(layers.Flatten())

# Fully connected 레이어
model.add(layers.Dense(512, activation='relu'))

# 출력 레이어 (이진 분류)
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

X,y = pilot_data['FC'], pilot_data['STATUS']


