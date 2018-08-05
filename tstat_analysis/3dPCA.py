#3dPCA.py


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn.cluster import KMeans
import os

import pandas as pd 
from sklearn import cluster, datasets


from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA, IncrementalPCA


label_file = os.path.join("controldata/cleanDataPCATest.csv")

df=pd.read_csv(label_file)


# load dataset into Pandas DataFrame
df = pd.read_csv(label_file, names=['Average rtt C2S','rtt min',
	'rtt max','max seg size','min seg size','win max','win min','cwin max',
	'cwin min','initial cwin','rtx RTO','rtx FR','reordering','unnece rtx RTO','target'])
print "*******"

features = ['Average rtt C2S','rtt min','rtt max','max seg size','min seg size',
'win max','win min','cwin max',
	'cwin min','initial cwin','rtx RTO','rtx FR','reordering','unnece rtx RTO']



# Separating out the features
X = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values

# Standardizing the features
X= StandardScaler().fit_transform(X)

print "1"
estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                               init='random'))]

fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Random', 0),
                    ('NoFlow', 1),
                    ('loss1%', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

fig.show()