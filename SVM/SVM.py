import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
import Functional_Connectivity
import Jun_data
import pandas as pd

jun_df = Jun_data.jun_df.apply(pd.to_numeric, errors='coerce')

X = np.array(Jun_data.jun_df['FC'].tolist())
y = np.array(Jun_data.jun_df['Status'].tolist())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = svm_model.predict(X_test)

print(y_train)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))