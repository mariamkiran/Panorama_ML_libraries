#tsne.py


# Importing Modules
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os
from sklearn.preprocessing import StandardScaler
# Loading dataset


label_file = os.path.join("controldata/cleanDataPCATest.csv")

#raw_data={'Average rtt C2S', 'Average rtt S2C','target'}
#df=pd.DataFrame(raw_data, columns = ['Sent','Received','Lost','Duplicated','Reordered'])

df=pd.read_csv(label_file)


# load dataset into Pandas DataFrame
df = pd.read_csv(label_file, names=['Average rtt C2S','rtt min',
	'rtt max','max seg size','min seg size','win max','win min','cwin max',
	'cwin min','initial cwin','rtx RTO','rtx FR','reordering','unnece rtx RTO','target'])
print "*******"
#print df

#extracting features
features = ['Average rtt C2S','rtt min','rtt max','max seg size','min seg size',
'win max','win min','cwin max',
	'cwin min','initial cwin','rtx RTO','rtx FR','reordering','unnece rtx RTO']

# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
print y
# Standardizing the features
x = StandardScaler().fit_transform(x)

# Defining Model
model = TSNE(learning_rate=100)

# Fitting Model
transformed = model.fit_transform(x)

# Plotting 2d t-Sne
x_axis = transformed[:, 0]
y_axis = transformed[:, 1]
targets = ['Random', 'Noflow','loss1%','loss5%', 'pDup1%', 'pDup5%','Reord25-50%','Reord50-50%']
colors = ['r', 'g', 'b', 'black', 'lime', 'yellow', 'cyan', 'coral']




for target, color in zip(targets,colors):
	plt.scatter(x_axis, y_axis, c = color, s = 50)
	plt.legend(targets)

plt.show()