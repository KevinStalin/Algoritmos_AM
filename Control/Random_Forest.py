# Random Forest con Iris dataset
from sklearn.ensemble import RandomForestClassifier

#Importando metricas para evaluacion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
'exec(%matplotlib inline)'
from sklearn import datasets
from sklearn.datasets import load_iris


#Cargar dataset
iris = datasets.load_iris()

# print the label
print(iris.target_names)

# print the names of the features
print(iris.feature_names)

# Creando un DataFrame.
import pandas as pd
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
data.head()


from sklearn.model_selection import train_test_split

# VARIABLES DEPENDIENTES
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features

# VARIABLES INDEPENDIENTES
y=data['species']  # Labels

# Dividir Entranamiento y Prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

#Importar Modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

#Crear un Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Entrenamiento del modelo usando los conjuntos de datos
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn import metrics

# Metricas
matrizConfusion = metrics.confusion_matrix(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average=None)
accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred , average=None)
f1 = metrics.f1_score(y_test, y_pred, average=None)

print("Matriz de Confusion: \n",metrics.confusion_matrix(y_test, y_pred))
print("PrecisiÃ³n: ",metrics.precision_score(y_test, y_pred, average=None))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred , average=None))
print("F1: ",metrics.f1_score(y_test, y_pred, average=None))

# Random Forest con Wine dataset
from sklearn.ensemble import RandomForestClassifier

#Importando metricas para evaluacion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
'exec(%matplotlib inline)'

from sklearn.datasets import load_wine

##DATA WINE
wine = datasets.load_wine()
df=pd.DataFrame(wine['data'])
# print(wine)

# print the label
print(wine.target_names)

# print the names of the features
print(wine.feature_names)

# 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
# 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
# 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'

# Creando un DataFrame
import pandas as pd
data=pd.DataFrame({
    'alcohol':wine.data[:,0],
    'malic_acid':wine.data[:,1],
    'ash':wine.data[:,2],
    'alcalinity_of_ash':wine.data[:,3],
    'magnesium':wine.data[:,4],
    'total_phenols':wine.data[:,5],
    'flavanoids':wine.data[:,6],
    'nonflavanoid_phenols':wine.data[:,7],
    'proanthocyanins':wine.data[:,8],
    'color_intensity':wine.data[:,9],
    'hue':wine.data[:,10],
    'od280/od315_of_diluted_wines':wine.data[:,11],
    'proline':wine.data[:,12],
    'target_names':wine.target
})
data.head()

# VARIABLES DEPENDIENTES
X=data[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']]  # Features

# VARIABLES INDEPENDIENTES
y=data['target_names']  # Labels

# Entrenamiento y Prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12) # 70% training and 30% test
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))

#Importar Modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

#Crear un Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Entrenamiento del modelo usando los conjuntos de datos
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn import metrics

matrizConfusion = metrics.confusion_matrix(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average=None)
accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred , average=None)
f1 = metrics.f1_score(y_test, y_pred, average=None)

print("Matriz de Confusion: \n",metrics.confusion_matrix(y_test, y_pred))
print("PrecisiÃ³n: ",metrics.precision_score(y_test, y_pred, average=None))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred , average=None))
print("F1: ",metrics.f1_score(y_test, y_pred, average=None))