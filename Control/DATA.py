# carga de los Dstaset
from sklearn import datasets

# carga en variables

iris = datasets.load_iris();
wine = datasets.load_wine();

nombres_iris = iris['target_names']
nombres_wine = wine['target_names']
