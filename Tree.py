import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from IPython.display import Image
from sklearn import tree
from subprocess import call

col_names = ["region", "city", "hotels", "lon", "lat", "turism"]
col_type = {'region': object, 'city': object, 'hotels': np.int32, 'lon': np.float64, 'lat': np.float64, 'turism': object}
data = pd.read_csv(os.getcwd()+"/../DecisionTree/dataset.csv", names=col_names, dtype=col_type)

# Trasformiamo le nostre categorie "Regioni" in valori così da poterle usare per elaborare i dati.
# Esse verranno rappresentate ciascuna da una colonna con un valore binario, 0 se non sarà appartenente a quella regione, 1 se invece vi appartiene.
data_tree = pd.get_dummies(data, columns=["region", "turism"])

# creazione del training set
data_tree = data_tree.drop(["city", "hotels"], axis=1)
x = data_tree
y = data["hotels"]

tree_clf = tree.DecisionTreeClassifier(criterion='gini', max_features=None, presort=False, splitter='best')
tree_clf = tree_clf.fit(x, y)

tree.export_graphviz(tree_clf, out_file="tree2.dot", max_depth=4, feature_names=None, rounded=True, precision=2,
                     filled=True, class_names=True)
call(['dot', '-Tpng', 'tree2.dot', '-o', 'tree2.png'])
Image(filename='tree2.png')

lat = input("Quale è la Latitudine desiderata?")
lon = input("Quale è la Longitudine desiderata?")
turismo = input("Quale è la tipologia di turismo desiderato?(Montagna,Lago,Mare,Arte)")


def type_of_turism(turismo):
    return {
        "Montagna": [1, 0, 0, 0],
        "Lago": [0, 1, 0, 0],
        "Mare": [0, 0, 1, 0],
        "Arte": [0, 0, 0, 1],
    }.get(turismo, "Tipologia di turismo non trovata.")

example = [lat, lon, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
example1 = example + type_of_turism(turismo)
risultati = tree_clf.predict([example1])
risultati = tree_clf.predict([example1])
print(risultati)
