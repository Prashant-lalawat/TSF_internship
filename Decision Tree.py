print(".....Decision Tree.....")
#Descison tree: it is a tree that has flowchart like structure, and it is used, and it is used for
#prediction and classification of data.

print("....Algorithm.....")
### Importing all libraries required
### Load the iris dataset
### Forming the iris dataframe
### Defining the decision tree algorithm
### Import necessary libraries for graph viz
### Visualize the graph

print("............CODE.............")
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris

iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
print(df)

df1 = df.rename(columns = {0:'sepal length(cm)', 1: 'sepal width(cm)', 2: 'petal length (cm)',
                           3: 'petal width (cm)'})
print(df1.head(150))

print(df1.info())
print(df.describe())

x = iris.target
print(x)

from sklearn import tree
Dtree = tree.DecisionTreeClassifier()
Dtree = Dtree.fit(df, x)
print(Dtree)

import graphviz
dot_data = tree.export_graphviz(Dtree, out_file=None, feature_names=iris.feature_names,
                                class_names=iris.target_names, filled = True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
print(graph)