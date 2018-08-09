#autoencoder.py

from __future__ import print_function

from sklearn import cluster, datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os

import tensorflow

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA, IncrementalPCA



#----
from keras.layers import Input, Dense
from keras.models import Model



from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.layers import Input

#----



def plot3clusters(X, title, vtitle):
  plt.figure()
  #colors = ['navy', 'turquoise', 'darkorange']
  targets = ['Normal', 'Loss1%','Loss5%', 'pDup1%', 'pDup5%','reord25-50%','reord50-50%', 'Loss3%']
  colors = ['r', 'g', 'b', 'black', 'lime', 'yellow', 'cyan', 'coral']
  lw = 2

  for color, i, target in zip(colors,[0,1,2,3,4,5,6,7], targets):
  	plt.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=1., lw=lw, label=target)
  
  plt.legend(loc='best', shadow=False, scatterpoints=1)
  plt.title(title)  
  plt.xlabel(vtitle + "1")
  plt.ylabel(vtitle + "2")
  plt.show()


#raw_data={'Average rtt C2S', 'Average rtt S2C','target'}
#df=pd.DataFrame(raw_data, columns = ['Sent','Received','Lost','Duplicated','Reordered'])

label_file = os.path.join("1000Genomelogs_labelleddata/fullfeature-26variables.csv")

df=pd.read_csv(label_file)


# load dataset into Pandas DataFrame
df = pd.read_csv(label_file, names=['Average rtt C2S', 'Average rtt S2C',
	'max seg size1','min seg size1','win max1','win min1','win zero1','cwin max1',
	'cwin min1','initial cwin1','rtx RTO1','rtx FR1','reordering1','net dup1','max seg size',
	'in seg size','win max','win min','win zero','cwin max','cwin min','initial cwin','rtx RTO','rtx FR',
	'eordering','net dup','target'])

print("*******")
#print df

#extracting features
features = ['Average rtt C2S', 'Average rtt S2C','max seg size1','min seg size1','win max1',
'win min1','win zero1','cwin max1','cwin min1','initial cwin1','rtx RTO1','rtx FR1','reordering1','net dup1',
'max seg size','in seg size','win max','win min','win zero','cwin max','cwin min','initial cwin','rtx RTO','rtx FR',
'eordering','net dup']


# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
print(y)
# Standardizing the features
x = StandardScaler().fit_transform(x)

print("1")

targets = ['Normal', 'Loss1%','Loss5%', 'pDup1%', 'pDup5%','reord25-50%','reord50-50%', 'Loss3%']
  

colors = ['r', 'g', 'b', 'black', 'lime', 'yellow', 'cyan', 'coral']


#create an AE and fit it with our data using 3 neurons in the dense layer using keras' functional API
input_dim =x.shape[1]
encoding_dim = 2  
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='linear')(input_img)
decoded = Dense(input_dim, activation='linear')(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

history = autoencoder.fit(x, x,
                epochs=1000,
                batch_size=16,
                shuffle=True,
                validation_split=0.1,
                verbose = 0)

print(history)
#plot our loss 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# use our encoded layer to encode the training input
# encoder = Model(input_img, encoded)
# encoded_input = Input(shape=(encoding_dim,))
# decoder_layer = autoencoder.layers[-1]
# decoder = Model(encoded_input, decoder_layer(encoded_input))
# encoded_data = encoder.predict(x)

# plot3clusters(encoded_data[:,:2], 'Linear AE', 'AE')  

# plt.show()