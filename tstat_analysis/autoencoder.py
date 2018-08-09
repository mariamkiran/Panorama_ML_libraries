#autoencoder.py

from pandas import read_csv, DataFrame
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


#----



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


 
# SCALE EACH FEATURE INTO [0, 1] RANGE
sX = minmax_scale(x, axis = 0)
ncol = sX.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(sX, y, train_size = 0.5, random_state = seed(2017))
 
### AN EXAMPLE OF SIMPLE AUTOENCODER ###
# InputLayer (None, 10)
#      Dense (None, 5)
#      Dense (None, 10)
print("training")
print(X_train)

print("test")
print(X_test)

input_dim = Input(shape = (ncol, ))
# DEFINE THE DIMENSION OF ENCODER ASSUMED 3
encoding_dim = 3
# DEFINE THE ENCODER LAYER
encoded = Dense(encoding_dim, activation = 'relu')(input_dim)
# DEFINE THE DECODER LAYER
decoded = Dense(ncol, activation = 'sigmoid')(encoded)
# COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
autoencoder = Model(input = input_dim, output = decoded)
# CONFIGURE AND TRAIN THE AUTOENCODER
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.fit(X_train, X_train, nb_epoch = 50, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))

# THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
encoder = Model(inputs = input_dim, outputs = encoded)
encoded_input = Input(shape = (encoding_dim, ))
encoded_out = encoder.predict(X_test)
encoded_out[0:2]

print(encoded_out)
#array([[ 0.        ,  1.26510417,  1.62803197],
#       [ 2.32508397,  0.99735016,  2.06461048]], dtype=float32)

