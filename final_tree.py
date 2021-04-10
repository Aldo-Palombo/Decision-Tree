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
# ----------------------------------------------------------------------------------------------------------------------
col_names = ["region", "city", "hotels", "lon", "lat", "turism"]
data = pd.read_csv(os.getcwd()+"/../DecisionTree/dataset.csv", names=col_names)

ricerca = input("Che tipo di ricerca vorresti fare?(citta,turismo,hotel)")

if ricerca == "citta":
    data_tree = pd.get_dummies(data, columns=["region", "turism"])
    data_tree[0:4]

    data_tree = data_tree.drop(["city", "hotels"], axis=1)
    x = data_tree
    y = data["city"]

    tree_clf = tree.DecisionTreeClassifier(criterion='gini', max_features=None, presort=False, splitter='best')
    tree_clf = tree_clf.fit(x, y)

    tree.export_graphviz(tree_clf, out_file="tree.dot", max_depth=4, feature_names=None, rounded=True,
                         precision=2, filled=True, class_names=True)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])
    Image(filename='tree.png')

    lat = input("Quale è la Latitudine desiderata?")
    lon = input("Quale è la Longitudine desiderata?")
    turismo = input("Quale è la tipologia di turismo desiderato?(Montagna,Lago,Mare,Arte)")

    def type_of_turism(turismo):
        return {
            "Montagna": [1, 0, 0, 0],
            "Lago": [0, 1, 0, 0],
            "Mare": [0, 0, 1, 0],
            "Arte": [0, 0, 0, 1],
        }.get(turismo,"Tipologia di turismo non trovata.")

    # latitudine, longitudine, turismo
    example = [lat, lon, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    example1 = example + type_of_turism(turismo)
    risultati = tree_clf.predict([example1])
    print(risultati)

if ricerca == "turismo":
    data_tree = pd.get_dummies(data, columns=["region"])
    data_tree[0:21]

    data_tree = data_tree.drop(["city", "turism"], axis=1)
    x = data_tree
    y = data["turism"]

    tree_clf = tree.DecisionTreeClassifier(criterion='gini', max_features=None, presort=False, splitter='best')
    tree_clf = tree_clf.fit(x, y)

    img = tree.export_graphviz(tree_clf, out_file="tree1.dot", feature_names=data_tree.columns, rounded=True,
                               filled=True)
    call(['dot', '-Tpng', 'tree1.dot', '-o', 'tree1.png'])
    Image(filename='tree1.png')

    hotel = input("Inserire il numero dell'hotel: ")
    lat = input("Inserire la latitudine: ")
    lon = input("Inserire la longitudine: ")
    regione = input("Quale è la regione da selezionare? ")

    def numbers_to_month(regione):
        return {
            "Valle d'aosta": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Piemonte": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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

    # hotel, latitudine, longitudine, regione
    example = [hotel, lat, lon]
    example1 = example + numbers_to_month(regione)
    risultati = tree_clf.predict([example1])
    print("La tipologia di turismo consigliato è: " + risultati)

if ricerca=="hotel":
    data_tree = pd.get_dummies(data, columns=["region", "turism"])

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
    print(risultati)
