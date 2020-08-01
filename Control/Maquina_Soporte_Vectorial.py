from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn import datasets

iris = datasets.load_iris();
wine = datasets.load_wine();
# print(list(wine['data']))
nombres_iris = iris['target_names']
nombres_wine = wine['target_names']

X_iris = np.array(iris['data'])
y_iris = np.array(iris['target'])

labels_I=[]
for i in y_iris:
    if i==0:
        labels_I.append('setosa')
    if i==1:
        labels_I.append('versicolor')
    if i==2:
        labels_I.append('virginica')

# X_trainI, X_testI, y_trainI, y_testI = train_test_split(XI, yI, test_size=0.3, random_state=12)
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=12)

algoritmoI = SVC()
algoritmoI.fit(X_train_iris, y_train_iris)
iris_predic = algoritmoI.predict(X_test_iris)

M_confusion_iris = metrics.confusion_matrix(y_test_iris, iris_predic)
Acurracy_iris = metrics.accuracy_score(y_test_iris, iris_predic)
Precision_iris = metrics.precision_score(y_test_iris, iris_predic, average=None)
Recall_iris = metrics.recall_score(y_test_iris, iris_predic, average=None)
F1_iris = metrics.f1_score(y_test_iris, iris_predic, average=None)

m = np.array(M_confusion_iris)
b = np.asarray(m)
# salida = np.sum(m, axis=1)
sum1 = m.sum(axis=0)
totalErrores_iris = sum(sum1) - np.trace(b)

Training_I = X_train_iris.shape[0]
Test_I = X_test_iris.shape[0]

print("*********************DATA WINE**************************")
X_wine = np.array(wine['data'])
y_wine = np.array(wine['target'])

labels_W=[]
for i in y_wine:
    if i==0:
        labels_W.append('class_0')
    if i==1:
        labels_W.append('class_1')
    if i==2:
        labels_W.append('class_2')

X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.2, random_state=12)
# X_trainw, X_testw, y_trainw, y_testw = train_test_split(Xw, yw, test_size=0.2, random_state=12)

algoritmoW = SVC()
algoritmoW.fit(X_train_wine, y_train_wine)
wine_predic = algoritmoW.predict(X_test_wine)

M_confusion_wine = metrics.confusion_matrix(y_test_wine, wine_predic)
Acurracy_wine = metrics.accuracy_score(y_test_wine, wine_predic)
Precision_wine = metrics.precision_score(y_test_wine, wine_predic, average=None)
Recall_wine = metrics.recall_score(y_test_wine, wine_predic, average=None)
F1_wine = metrics.f1_score(y_test_wine, wine_predic, average=None)

m = np.array(M_confusion_wine)
b = np.asarray(m)
# salida = np.sum(m, axis=1)
sum1 = m.sum(axis=0)
totalErrores_wine = sum(sum1) - np.trace(b)

Training_W = X_train_wine.shape[0]
Test_W = X_test_wine.shape[0]