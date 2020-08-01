import DATA as data
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

X = np.array(data.iris['data'])
y = np.array(data.iris['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=12)

logistic=KNeighborsClassifier()
logistic.fit(X_train,y_train)
y_pred5=logistic.predict(X_test)

# Matriz confusion
# print(confusion_matrix(y_test,y_pred5))
matrix_aux = confusion_matrix(y_test, y_pred5)
##Medidas
m = np.array(matrix_aux)
b = np.asarray(m)
array = m
salida = np.sum(array, axis=1)
sum1 = m.sum(axis=0)
FP = m.sum(axis=0) - np.diag(m)
FN = m.sum(axis=1) - np.diag(m)
TP = np.diag(m)
TN = m.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

#precision
PPV = TP / (TP + FP)
#recall
TPR = TP / (TP + FN)
# F-Measure
F1 = 2 * (PPV * TPR) / (PPV + TPR)
#accuracy
ACCC = np.trace(b) / sum(sum1)
#total de rrores
totalErrores = sum(sum1) - np.trace(b)

print(classification_report(y_test,y_pred5))
# =================================================================================================
#   Wine
# =================================================================================================

X2 = np.array(data.wine['data'])
y2 = np.array(data.wine['target'])

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.4, random_state=12)

logistic2 = KNeighborsClassifier()
logistic2.fit(X_train, y_train)
y_pred6 = logistic2.predict(X_test)

# Matriz confusion
# print(confusion_matrix(y_test,y_pred5))
matrix_aux2 = confusion_matrix(y_test, y_pred6)
##Medidas
m2 = np.array(matrix_aux2)
b = np.asarray(m2)
array = m2
salida = np.sum(array, axis=1)
sum1 = m2.sum(axis=0)
FP = m2.sum(axis=0) - np.diag(m2)
FN = m2.sum(axis=1) - np.diag(m2)
TP = np.diag(m2)
TN = m2.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# precision
PPV2 = TP / (TP + FP)
# recall
TPR2 = TP / (TP + FN)
# F-Measure
F12 = 2 * (PPV2 * TPR2) / (PPV2 + TPR2)
# accuracy
ACCC2 = np.trace(b) / sum(sum1)
# total de rrores
totalErrores2 = sum(sum1) - np.trace(b)

print(classification_report(y_test, y_pred6))
