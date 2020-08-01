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
#Importando metricas para evaluacion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
'exec(%matplotlib inline)'
#print(data.iris)
#print(data.wine)
##DATA IRIS
iris = data.iris
df=pd.DataFrame(iris['data'])
#iris = datasets.load_iris()
#df=pd.DataFrame(iris['data'])
#house=datasets.load_boston()
#df1=pd.DataFrame(house['data'])
#print(iris)
#print(df1)
##VARIABLES INDEPENDIENTESS
X=(np.array(iris['data']))
#print("variables independientes",X)
##VARIABLES DEPENDIENTES
y = np.array(iris['target'])
#print("variables dependientes", y)
#Modelo de Máquinas de Vectores de Soporte
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))
algoritmo = SVC()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisión Máquinas de Vectores de Soporte (data de iris): {}'.format(algoritmo.score(X_train, y_train)))
##MATRIZ
print(classification_report(y_test,Y_pred))
print("MATRIZ DE CONFUSION")
print(confusion_matrix(y_test,Y_pred))
##Precision
#print('accurancy de iris es: ', accuracy_score(Y_pred,y_test))
#print("********************************************************************")
print("\n")
print("Accuracy:",metrics.accuracy_score(y_test,Y_pred))
print("Precisión: ",metrics.precision_score(y_test,Y_pred, average=None))
print("Recall: ",metrics.recall_score(y_test,Y_pred , average=None))
print("F1: ",metrics.f1_score(y_test,Y_pred, average=None))
print("**////////////////////////////////////////////////////////////////////////")
##/////////////////////////////////////////////////////////////////////////////////
##data.wine
##/////////////////////////////////////////////////////////////////////////////////
dfvin=pd.DataFrame(data.wine['data'])
#iris = datasets.load_iris()
#df=pd.DataFrame(iris['data'])
#house=datasets.load_boston()
#df1=pd.DataFrame(house['data'])
#print(iris)
#print(df1)
##VARIABLES INDEPENDIENTESS
Xvin=(np.array(data.wine['data']))
#print("variables independientes",X)
##VARIABLES DEPENDIENTES
yvin = np.array(data.wine['target'])
#print("variables dependientes", y)
#Modelo de Máquinas de Vectores de Soporte
X_traink, X_testk, y_traink, y_testk = train_test_split(Xvin, yvin, test_size=0.2, random_state=12)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_traink.shape[0], X_testk.shape[0]))
algoritmo = SVC()
algoritmo.fit(X_traink, y_traink)
Y_predk = algoritmo.predict(X_testk)
print('Precisión Máquinas de Vectores de Soporte (data de vinos): {}'.format(algoritmo.score(X_traink, y_traink)))
##MATRIZ
print(classification_report(y_testk,Y_predk))

print("MATRIZ DE CONFUSION")
print(confusion_matrix(y_testk,Y_predk))
##Precision
#print('accurancy de vinos es: ', accuracy_score(Y_predk,y_testk))
print("\n")
print("Accuracy:",metrics.accuracy_score(y_testk,Y_predk))
print("Precisión: ",metrics.precision_score(y_testk,Y_predk, average=None))
print("Recall: ",metrics.recall_score(y_testk,Y_predk , average=None))
print("F1: ",metrics.f1_score(y_testk,Y_predk, average=None))

