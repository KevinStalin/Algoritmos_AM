import DATA as data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
X = np.array(data.iris['data'])
#print("##",X)
##VARIABLES DEPENDIENTES
y = np.array(data.iris['target'])
#Modelo de Máquinas de Vectores de Soporte
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))

algoritmo = SVC()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisión Máquinas de Vectores de Soporte: {}'.format(algoritmo.score(X_train, y_train)))
#######