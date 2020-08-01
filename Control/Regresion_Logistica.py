import DATA as data
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import metrics

logistic = KNeighborsClassifier()

# =================================================================================================
#   IRIS
# =================================================================================================
X_iris = np.array(data.iris['data'])
y_iris = np.array(data.iris['target'])

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.4, random_state=12)

logistic.fit(X_train_iris, y_train_iris)
iris_predic = logistic.predict(X_test_iris)

M_confusion_iris = metrics.confusion_matrix(y_test_iris, iris_predic)
Acurracy_iris = metrics.accuracy_score(y_test_iris, iris_predic)
Precision_iris = metrics.precision_score(y_test_iris, iris_predic, average=None)
Recall_iris = metrics.recall_score(y_test_iris, iris_predic, average=None)
F1_iris = metrics.f1_score(y_test_iris, iris_predic, average=None)


# =================================================================================================
#   Wine
# =================================================================================================
X_wine = np.array(data.wine['data'])
y_wine = np.array(data.wine['target'])

X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.4, random_state=12)

logistic.fit(X_train_wine, y_train_wine)
wine_predic = logistic.predict(X_test_wine)

M_confusion_wine = metrics.confusion_matrix(y_test_wine, wine_predic)
Acurracy_wine = metrics.accuracy_score(y_test_wine, wine_predic)
Precision_wine = metrics.precision_score(y_test_wine, wine_predic, average=None)
Recall_wine = metrics.recall_score(y_test_wine, wine_predic, average=None)
F1_wine = metrics.f1_score(y_test_wine, wine_predic, average=None)
