import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, layers

# 매번 동일한 결과를 가져 오기 위함.난수를 생성 하는 코드를 실행 하면 항상 같은 난수가 형성됨.
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

X_test, y_test = test_images, test_labels

print(len(y_test))