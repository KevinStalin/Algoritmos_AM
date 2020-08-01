import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import datasets

# carga en variables

iris = datasets.load_iris();
wine = datasets.load_wine();
classifiern = GaussianNB()

# =================================================================================================
#   IRIS
# =================================================================================================
X_iris = np.array(iris['data'])
y_iris = np.array(iris['target'])

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.4, random_state=12)

classifiern.fit(X_train_iris, y_train_iris)
iris_predic = classifiern.predict(X_test_iris)

M_confusion_iris = metrics.confusion_matrix(y_test_iris, iris_predic)
Acurracy_iris = metrics.accuracy_score(y_test_iris, iris_predic)
Precision_iris = metrics.precision_score(y_test_iris, iris_predic, average=None)
Recall_iris = metrics.recall_score(y_test_iris, iris_predic, average=None)
F1_iris = metrics.f1_score(y_test_iris, iris_predic, average=None)

m = np.array(M_confusion_iris)
b = np.asarray(m)
salida = np.sum(m, axis=1)
sum1 = m.sum(axis=0)
totalErrores_iris = sum(sum1) - np.trace(b)

Training = X_train_iris.shape[0]
Test = X_test_iris.shape[0]
# =================================================================================================
#   Wine
# =================================================================================================
X_wine = np.array(wine['data'])
y_wine = np.array(wine['target'])

X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.4, random_state=12)

classifiern.fit(X_train_wine, y_train_wine)
wine_predic = classifiern.predict(X_test_wine)

M_confusion_wine = metrics.confusion_matrix(y_test_wine, wine_predic)
Acurracy_wine = metrics.accuracy_score(y_test_wine, wine_predic)
Precision_wine = metrics.precision_score(y_test_wine, wine_predic, average=None)
Recall_wine = metrics.recall_score(y_test_wine, wine_predic, average=None)
F1_wine = metrics.f1_score(y_test_wine, wine_predic, average=None)

m = np.array(M_confusion_wine)
b = np.asarray(m)
salida = np.sum(m, axis=1)
sum1 = m.sum(axis=0)
totalErrores_wine = sum(sum1) - np.trace(b)
