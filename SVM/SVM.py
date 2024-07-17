import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import Functional_Connectivity

RBD = Functional_Connectivity.correlation_matrix_RBD
HC = Functional_Connectivity.correlation_matrix_HC

X = np.array([RBD, HC])
y = np.array([1, 0])

X = np.array([X.flatten() for i in X])

model = svm.SVC(kernel='linear')
model.fit(X, y)

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print(accuracy)


