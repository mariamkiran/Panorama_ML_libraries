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


label_file = os.path.join("exp1pegasus_full_filtered.csv")
#fullfeaturedata-windows.csv")

#raw_data={'Average rtt C2S', 'Average rtt S2C','target'}
#df=pd.DataFrame(raw_data, columns = ['Sent','Received','Lost','Duplicated','Reordered'])

df=pd.read_csv(label_file)


# load dataset into Pandas DataFrame
df = pd.read_csv(label_file, names=['Average rtt C2S', 'Average rtt S2C','max seg size1','min seg size1','win max1','win min1','win zero1','cwin max1','cwin min1','initial cwin1','rtx RTO1','rtx FR1','reordering1','net dup1','max seg size','in seg size','win max','win min','win zero','cwin max','cwin min','initial cwin','rtx RTO','rtx FR','eordering','net dup','target'])

print "*******"
print df

#extracting features
features = ['Average rtt C2S', 'Average rtt S2C',
'max seg size1',
'win min1',
'win zero1','cwin max1',
'cwin min1','initial cwin1',
'rtx RTO1','rtx FR1',
'reordering1',
'max seg size','in seg size',
'win max',
'cwin max',
'cwin min','initial cwin',
'rtx RTO']


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

