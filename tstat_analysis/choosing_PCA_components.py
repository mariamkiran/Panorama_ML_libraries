#chhosing PCA

from sklearn import cluster, datasets
import matplotlib.pyplot as plt
import pandas as pd 
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly

import plotly.plotly as py
plotly.__version__

from plotly.graph_objs import Bar, Scatter, Data, YAxis, Layout, Figure


label_file = os.path.join("controldata/cleanExp1_cleanfeatures.csv")
#fullfeaturedata-windows.csv")

#raw_data={'Average rtt C2S', 'Average rtt S2C','target'}
#df=pd.DataFrame(raw_data, columns = ['Sent','Received','Lost','Duplicated','Reordered'])

df=pd.read_csv(label_file)


# load dataset into Pandas DataFrame
#df = pd.read_csv(label_file, names=['Average rtt C2S', 'Average rtt S2C','max seg size1','min seg size1','win max1','win min1','win zero1','cwin max1','cwin min1','initial cwin1','rtx RTO1','rtx FR1','reordering1','net dup1','max seg size','in seg size','win max','win min','win zero','cwin max','cwin min','initial cwin','rtx RTO','rtx FR','eordering','net dup','target'])
df = pd.read_csv(label_file, names=['packets','RST sent','ACK sent','PURE ACK sent',
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
 ,'httpignore8','target'])


print "*******"
print df

#extracting features
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
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
X_std = StandardScaler().fit_transform(x)


mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)


print('\nEigenvalues \n%s' %eig_vals)

cor_mat1 = np.corrcoef(X_std.T)

#eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

#print('Eigenvectors iof correlation matrix \n%s' %eig_vecs)
#print('\nEigenvalues of correlation matrix \n%s' %eig_vals)


#cor_mat2 = np.corrcoef(X.T)

#eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

#print('Eigenvectors iof correlation matrix  2\n%s' %eig_vecs)
#print('\nEigenvalues iof correlation matrix 2\n%s' %eig_vals)

#for ev in eig_vecs:
 #   np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
#print('Everything ok!')

# Make a list of (eigenvalue, eigenvector) tuples
#eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
#eig_pairs.sort()
#eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
#print('Eigenvalues in descending order:')
#for i in eig_pairs:
 #   print(i[0])


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

trace1 = Bar(
        x=['PC %s' %i for i in range(1,5)],
        y=var_exp,
        showlegend=False)

trace2 = Scatter(
        x=['PC %s' %i for i in range(1,5)], 
        y=cum_var_exp,
        name='cumulative explained variance')

data = Data([trace1, trace2])

layout=Layout(
        yaxis=YAxis(title='Explained variance in percent'),
        title='Explained variance by different principal components')

plotly.offline.plot({
    "data": Data([trace1, trace2]),
    "layout": Layout(
        yaxis=YAxis(title='Explained variance in percent'),
        title='Explained variance by different principal components')
})

