import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from Data.pilot_data_HC import pilot_data
from sklearn import metrics as mt

data = pilot_data

X = np.array([x.flatten() for x in pilot_data['FC']])
y = np.array(pilot_data['STATUS'])

#k_fold 교차검증 실시할것(cross_validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = svm.SVC(kernel='linear')

model.fit(X_train, y_train)

predicted_result = model.predict(X_test)


scores = mt.classification_report(y_test, predicted_result)

print(scores)

