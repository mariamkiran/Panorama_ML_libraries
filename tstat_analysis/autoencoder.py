#autoencoder.py

from __future__ import print_function

from sklearn import cluster, datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os


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
label_file = os.path.join("controldata/cleanDataPCATest.csv")

#raw_data={'Average rtt C2S', 'Average rtt S2C','target'}
#df=pd.DataFrame(raw_data, columns = ['Sent','Received','Lost','Duplicated','Reordered'])

df=pd.read_csv(label_file)


# load dataset into Pandas DataFrame
df = pd.read_csv(label_file, names=['Average rtt C2S','rtt min',
	'rtt max','max seg size','min seg size','win max','win min','cwin max',
	'cwin min','initial cwin','rtx RTO','rtx FR','reordering','unnece rtx RTO','target'])

print("*******")
#print df

#extracting features
features = ['Average rtt C2S','rtt min','rtt max','max seg size','min seg size',
'win max','win min','cwin max',
	'cwin min','initial cwin','rtx RTO','rtx FR','reordering','unnece rtx RTO']



# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
print(y)
# Standardizing the features
x = StandardScaler().fit_transform(x)

print("1")

targets = ['Random', 'Noflow','loss1%','loss5%', 'pDup1%', 'pDup5%','Reord25-50%','Reord50-50%']


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

#plot our loss 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# use our encoded layer to encode the training input
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
encoded_data = encoder.predict(x)

plot3clusters(encoded_data[:,:2], 'Linear AE', 'AE')  


