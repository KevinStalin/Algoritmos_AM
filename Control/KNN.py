import DATA as data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import metrics
# Importando metricas para evaluacion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

iris = data.iris
df = pd.DataFrame(iris['data'])
X = (np.array(iris['data']))
y = np.array(iris['target'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
algoritmo = SVC()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)

M_confusion_iris = metrics.confusion_matrix(y_test, Y_pred)
Acurracy_iris = metrics.accuracy_score(y_test, Y_pred)
Precision_iris = metrics.precision_score(y_test, Y_pred, average=None)
Recall_iris = metrics.recall_score(y_test, Y_pred, average=None)
F1_iris = metrics.f1_score(y_test, Y_pred, average=None)

m = np.array(M_confusion_iris)
b = np.asarray(m)
salida = np.sum(m, axis=1)
sum1 = m.sum(axis=0)
totalErrores_iris = sum(sum1) - np.trace(b)

Training = X_train.shape[0]
Test = X_test.shape[0]
##/////////////////////////////////////////////////////////////////////////////////
##data.wine
##/////////////////////////////////////////////////////////////////////////////////

Xvin = (np.array(data.wine['data']))
yvin = np.array(data.wine['target'])

X_traink, X_testk, y_traink, y_testk = train_test_split(Xvin, yvin, test_size=0.2, random_state=12)

algoritmo = SVC()
algoritmo.fit(X_traink, y_traink)
Y_predk = algoritmo.predict(X_testk)

M_confusion_wine = metrics.confusion_matrix(y_testk, Y_predk)
Acurracy_wine = metrics.accuracy_score(y_testk, Y_predk)
Precision_wine = metrics.precision_score(y_testk, Y_predk, average=None)
Recall_wine = metrics.recall_score(y_testk, Y_predk, average=None)
F1_wine = metrics.f1_score(y_testk, Y_predk, average=None)

m = np.array(M_confusion_wine)
b = np.asarray(m)
salida = np.sum(m, axis=1)
sum1 = m.sum(axis=0)
totalErrores_wine = sum(sum1) - np.trace(b)

# Training_I = X_traink.shape[0]
# Test_I = X_testk.shape[0]
