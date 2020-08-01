# Random Forest con Iris dataset
from sklearn.ensemble import RandomForestClassifier

# Importando metricas para evaluacion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

'exec(%matplotlib inline)'
from sklearn import datasets
from sklearn.datasets import load_iris

# Cargar dataset
iris = datasets.load_iris()

print("--------------- Random Forest con Iris Data Set ---------------")
print("--------------- Clases Iris ---------------")
# print the label
print(iris.target_names)

print("--------------- Caracterisiticas Iris ---------------")
# print the names of the features
print(iris.feature_names)

# Creando un DataFrame.
import pandas as pd

data = pd.DataFrame({
'sepal length': iris.data[:, 0],
'sepal width': iris.data[:, 1],
'petal length': iris.data[:, 2],
'petal width': iris.data[:, 3],
'species': iris.target
})
data.head()

from sklearn.model_selection import train_test_split

# VARIABLES DEPENDIENTES
X = data[['sepal length', 'sepal width', 'petal length', 'petal width']] # Features

# VARIABLES INDEPENDIENTES
y = data['species'] # Labels

# Dividir Entranamiento y Prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12) # 70% training and 30% test
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))

# Entrenamiento Iris
X_train_iris = X_train
y_train_iris = y_train

# Prueba Iris
X_test_iris = X_test
y_test_iris = y_test

# Importar Modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

# Crear un Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

# Entrenamiento del modelo usando los conjuntos de datos
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn import metrics

# Metricas
matrizConfusion = metrics.confusion_matrix(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average=None)
accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred, average=None)
f1 = metrics.f1_score(y_test, y_pred, average=None)
print("--------------- Rendimiento del dataset Iris ---------------")
print("Matriz de Confusion: \n", matrizConfusion)
print("Precision: ", precision)
print("Accuracy:", accuracy)
print("Recall: ", recall)
print("F1: ", f1)
print()

from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Por último, calculamos el error cuadrático medio y el estadístico R^2
print (u'Error cuadrático medio: %.2f' % mean_squared_error(y_test, y_pred))
print (u'Estadístico R_2: %.2f' % r2_score(y_test, y_pred))
from sklearn.metrics import mean_absolute_error
print('MEAN : ',mean_absolute_error(y_test, y_pred))
print('RMSE',mean_squared_error(y_test, y_pred, squared=False))
print('MSE',mean_squared_error(y_test, y_pred, squared=True))
print()
print('################### R2 #############')
r2=r2_score(y_test, y_pred)
print(r2)
print('################### R2a #############')
r2a = 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)
print(r2a)
print('################### R2 y Ra #############')
X1 = sm.add_constant(X)
result = sm.OLS(y, X1).fit()
#print dir(result)
print (result.rsquared, result.rsquared_adj)
print()

import numpy as np

m = np.array(matrizConfusion)
b = np.asarray(m)
salida = np.sum(m, axis=1)
sum1 = m.sum(axis=0)
totalErrores_iris = sum(sum1) - np.trace(b)
print('Total Errores Iris: ', totalErrores_iris)
print()

# Random Forest con Wine dataset
from sklearn.ensemble import RandomForestClassifier

# Importando metricas para evaluacion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

'exec(%matplotlib inline)'

from sklearn.datasets import load_wine

##DATA WINE
wine = datasets.load_wine()
df = pd.DataFrame(wine['data'])
# print(wine)
print("--------------- Random Forest con Wine Data Set ---------------")
print("--------------- Clases Wine ---------------")
# print the label
print(wine.target_names)

print("--------------- Caracteristicas Wine ---------------")
# print the names of the features
print(wine.feature_names)

# Creando un DataFrame
import pandas as pd

data2 = pd.DataFrame({
'alcohol': wine.data[:, 0],
'malic_acid': wine.data[:, 1],
'ash': wine.data[:, 2],
'alcalinity_of_ash': wine.data[:, 3],
'magnesium': wine.data[:, 4],
'total_phenols': wine.data[:, 5],
'flavanoids': wine.data[:, 6],
'nonflavanoid_phenols': wine.data[:, 7],
'proanthocyanins': wine.data[:, 8],
'color_intensity': wine.data[:, 9],
'hue': wine.data[:, 10],
'od280/od315_of_diluted_wines': wine.data[:, 11],
'proline': wine.data[:, 12],
'target_names': wine.target
})
data2.head()

# VARIABLES DEPENDIENTES
X2 = data2[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids',
'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines',
'proline']] # Features

# VARIABLES INDEPENDIENTES
y2 = data2['target_names'] # Labels

# Entrenamiento y Prueba
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=12) # 70% training and 30% test
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train2.shape[0], X_test2.shape[0]))

# Entrenamiento Wine
X_train_wine = X_train2
y_train_wine = y_train2

# Prueba Wine
X_test_wine = X_test2
y_test_wine = y_test2

# Importar Modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

# Crear un Gaussian Classifier
clf2 = RandomForestClassifier(n_estimators=100)

# Entrenamiento del modelo usando los conjuntos de datos
clf2.fit(X_train2, y_train2)

y_pred2 = clf2.predict(X_test2)

from sklearn import metrics

matrizConfusion2 = metrics.confusion_matrix(y_test2, y_pred2)
precision2 = metrics.precision_score(y_test2, y_pred2, average=None)
accuracy2 = metrics.accuracy_score(y_test2, y_pred2)
recall2 = metrics.recall_score(y_test2, y_pred2, average=None)
f1_2 = metrics.f1_score(y_test2, y_pred2, average=None)
print("--------------- Rendimiento del dataset Wine ---------------")
print("Matriz de Confusion: \n", matrizConfusion2)
print("Precision: ", precision2)
print("Accuracy:", accuracy2)
print("Recall: ", recall2)
print("F1: ", f1_2)
print()

from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Por último, calculamos el error cuadrático medio y el estadístico R^2
print (u'Error cuadrático medio: %.2f' % mean_squared_error(y_test2, y_pred2))
print (u'Estadístico R_2: %.2f' % r2_score(y_test2, y_pred2))
from sklearn.metrics import mean_absolute_error
print('MEAN : ',mean_absolute_error(y_test2, y_pred2))
print('RMSE',mean_squared_error(y_test2, y_pred2, squared=False))
print('MSE',mean_squared_error(y_test2, y_pred2, squared=True))
print()
print('################### R2 #############')
r22=r2_score(y_test2, y_pred2)
print(r22)
print('################### R2a #############')
r2a2 = 1 - (1-r2)*(len(y)-1)/(len(y)-X2.shape[1]-1)
print(r2a2)
print()

import numpy as np

m2 = np.array(matrizConfusion2)
b2 = np.asarray(m2)
salida2 = np.sum(m2, axis=1)
sum11 = m2.sum(axis=0)
totalErrores_wine = sum(sum11) - np.trace(b2)
print('Total Errores Wine: ', totalErrores_wine)
print()