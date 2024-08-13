import tensorflow as tf
from tensorflow.keras import layers, models

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

# Flatten 레이어로 1D 벡터로 변환
model.add(layers.Flatten())

# Fully connected 레이어
model.add(layers.Dense(512, activation='relu'))

# 출력 레이어 (이진 분류)
model.add(layers.Dense(1, activation='softmax'))

print(model.summary())
