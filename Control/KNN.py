from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import svm
# import pandas as pd
import numpy as np
from sklearn import metrics
# Importando metricas para evaluacion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

iris = datasets.load_iris();
wine = datasets.load_wine();
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
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.4, random_state=12)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

algoritmo = SVC()
algoritmo.fit(X_train_iris, y_train_iris)
# Y_pred = algoritmo.predict(X_test_iris)
iris_predic = algoritmo.predict(X_test_iris)

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
##/////////////////////////////////////////////////////////////////////////////////
##data.wine
##/////////////////////////////////////////////////////////////////////////////////

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

# X_traink, X_testk, y_traink, y_testk = train_test_split(Xvin, yvin, test_size=0.2, random_state=12)
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.4, random_state=12)

algoritmo = SVC()
algoritmo.fit(X_train_wine, y_train_wine)
wine_predic = algoritmo.predict(X_test_wine)

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