#kmeans

#finding normal and abnormal transfers in normal experiment
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd

plt.rcParams['figure.figsize'] = (16, 9)

# Creating a sample dataset with 4 clusters
dataset = pd.read_csv("1000Genomelogs_labelleddata/tcp_exp1_clean.csv")
#controldata/cleanExp1.csv")
#

features = ['packets','RST sent','ACK sent','PURE ACK sent',
'unique bytes','data pkts','data bytes','rexmit pkts','rexmit bytes',
'out seq pkts','SYN count','FIN count','packets2','RST sent2','ACK sent2',
'PURE ACK sent2','unique bytes2','data pkts2','data bytes2','rexmit pkts2','rexmit bytes2',
'out seq pkts2','SYN count2','FIN count2',
'Completion time','C first payload','S first payload','C last payload','S last payload',
 'C first ack','S first ack','C Internal','S Internal','C anonymized','S anonymized',
 'Connection type','P2P type','HTTP type','Average rtt C2S','rtt min','rtt max',
 'Stdev rtt','rtt count','ttl_min','ttl_max','Average rtt S2C','rtt min2','rtt max2','Stdev rtt2',
 'rtt count2','ttl_min2','ttl_max2','P2P subtype','ED2K Data',
 'ED2K Signaling','ED2K C2S','ED2K C2C','ED2K Chat','RFC1323 ws','RFC1323 ts',
 'window scale','SACK req','SACK sent','MSS','max seg size','min seg size',
 'win max','win min','win zero','cwin max','cwin min','initial cwin','rtx RTO',
 'rtx FR','reordering','net dup','unknown','flow control','unnece rtx RTO','unnece rtx FR',
 'SYN seqno','RFC1323 ws','RFC1323 ts','window scale',
 'SACK req','SACK sent','MSS2','max seg size2','min seg size2','win max2','win min2',
 'win zero2','cwin max2','cwin min2','initial cwin2','rtx RTO2','rtx FR2','reordering2',
 'net dup2','unknown2','flow control2','unnece rtx RTO2','unnece rtx FR2','SYN seqno2',
 'httpignore','httpignore2','httpignore3','httpignore4','httpignore5','httpignore6','httpignore7'
 ,'httpignore8']



# Separating out the features
x = dataset.loc[:, features].values

print(x)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    print(kmeans.inertia_)
    wcss.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
#plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()


#Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'green', label = 'Group 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'red', label = 'Group 2')
#plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'blue', label = 'Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()
plt.show()
print(kmeans.labels_)

