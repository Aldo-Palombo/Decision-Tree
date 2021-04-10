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
#-----------------------------------------------------------------------------------------------------------------------
col_names = ["region", "city", "hotels", "lon", "lat", "turism"]
#col_type = {'region': object, 'city': object, 'hotels': np.int32, 'lon': np.float64, 'lat': np.float64, 'turism': object}
data = pd.read_csv(os.getcwd()+"/../DecisionTree/dataset.csv", names=col_names) #, dtype=col_type)

#Trasformiamo le nostre categorie "Regioni" in valori così da poterle usare per elaborare i dati.
#Esse verranno rappresentate ciascuna da una colonna con un valore binario, 0 se non sarà appartenente a quella regione, 1 se invece vi appartiene.
data_tree = pd.get_dummies(data, columns= ["region"])
data_tree[0:21]

#creazione del training set
data_tree = data_tree.drop(["city", "turism"], axis=1)
x = data_tree
y = data["turism"]

tree_clf = tree.DecisionTreeClassifier(criterion='gini', max_features=None, presort=False, splitter='best')
tree_clf = tree_clf.fit(x, y)

#Il file grafico è generato da questa funzione. In output abbiamo un file dot, che deve essere trasformato in file PNG.
img = tree.export_graphviz(tree_clf, out_file="tree1.dot", feature_names=data_tree.columns, rounded=True, filled=True)
call(['dot', '-Tpng', 'tree1.dot', '-o', 'tree1.png'])
Image(filename='tree1.png')

hotel = input("Inserire il numero dell'hotel: ")
lat = input("Inserire la latitudine: ")
lon = input("Inserire la longitudine: ")
regione = input("Quale è la regione da selezionare? ")
def numbers_to_month(regione):
    return {
        "Valle d'aosta": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Piemonte":[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Liguria": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Lombardia": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Trentino": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Veneto": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Friuli Venezia Giulia": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Emilia Romagna": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Toscana": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Umbria": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Marche": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Lazio": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "Abruzzo": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "Molise": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "Campania": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "Puglia": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "Basilicata": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "Calabria": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "Sicilia": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "Sardegna": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }.get(regione, "Regione non trovata")

#hotel, latitudine, longitudine, regione
example = [hotel, lat, lon] #,numbers_to_month(regione)]
example1 = example + numbers_to_month(regione)
risultati = tree_clf.predict([example1])
print("La tipologia di turismo consigliato è: " + risultati)
