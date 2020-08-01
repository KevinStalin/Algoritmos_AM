import DATA as data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pandas as pd
import numpy as np

XI = np.array(data.iris['data'])
#print("##",X)
##VARIABLES DEPENDIENTES
yI = np.array(data.iris['target'])
#Modelo de Máquinas de Vectores de Soporte
X_trainI, X_testI, y_trainI, y_testI = train_test_split(XI, yI, test_size=0.3, random_state=12)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_trainI.shape[0], X_testI.shape[0]))

algoritmoI = SVC()
algoritmoI.fit(X_trainI, y_trainI)
y_predI = algoritmoI.predict(X_testI)
prescoreI=algoritmoI.score(X_trainI, y_trainI)

###Medidas de rendimiento

matrixconfusionI=metrics.confusion_matrix(y_testI, y_predI)
precisionI=metrics.precision_score(y_testI, y_predI, average=None)
accuracyI=metrics.accuracy_score(y_testI, y_predI)
recallI=metrics.recall_score(y_testI, y_predI , average=None)
f1I=metrics.f1_score(y_testI, y_predI, average=None)

m = np.array(matrixconfusionI)
b = np.asarray(m)
salida = np.sum(m, axis=1)
sum1 = m.sum(axis=0)
totalErrores_iris = sum(sum1) - np.trace(b)

'''
print('Precisión Máquinas de Vectores de Soporte: {}'.format(prescoreI))

print("Medidas de rendimiento ")
print("Matriz de Confusion: \n",matrixconfusionI)
print("Precisión: \n",precisionI)
print("Accuracy:\n",accuracyI)
print("Recall: \n",recallI)
print("F1: \n",f1I)
'''
###########

##DATA WINE

print("*********************DATA WINE**************************")
dfvin=pd.DataFrame(data.wine['data'])
##VARIABLES INDEPENDIENTESS
Xw = np.array(data.wine['data'])
##VARIABLES DEPENDIENTES
yw = np.array(data.wine['target'])
X_trainw, X_testw, y_trainw, y_testw = train_test_split(Xw, yw, test_size=0.2, random_state=12)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_trainw.shape[0], X_testw.shape[0]))
#Modelo de Máquina de Soporte Vectorial
algoritmoW = SVC()
algoritmoW.fit(X_trainw, y_trainw)
y_predW = algoritmoW.predict(X_testw)
prescoreW=algoritmoW.score(X_trainw, y_trainw)
print('Precisión Máquina de Soporte Vectorial: {}'.format(prescoreW))

# print("Medidas de rendimiento ")
matrixconfusionW=metrics.confusion_matrix(y_testw, y_predW)
precisionW=metrics.precision_score(y_testw, y_predW, average=None)
accuracyW=metrics.accuracy_score(y_testw, y_predW)
recallW=metrics.recall_score(y_testw, y_predW , average=None)
f1W=metrics.f1_score(y_testw, y_predW, average=None)

m = np.array(matrixconfusionW)
b = np.asarray(m)
salida = np.sum(m, axis=1)
sum1 = m.sum(axis=0)
totalErrores_wine = sum(sum1) - np.trace(b)
'''
print("Matriz de Confusion: \n",matrixconfusionW)
print("Precisión: \n",precisionW)
print("Accuracy:\n",accuracyW)
print("Recall: \n",recallW)
print("F1: \n",f1W)
'''