from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from __future__ import print_function
from IPython.display import SVG, Image
from keras import regularizers, Model
from matplotlib import rc

import keras
import matplotlib.pyplot as plt
import numpy as np

num_classes = 10
input_dim = 784
batch_size = 256

(x_train, y_train), (x_val, y_val) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.

x_train = np.concatenate((x_train, x_val))

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))
print(x_train.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

early_stopping = EarlyStopping(monitor='val_loss', patience=4)

# ======================================================================================================================
# train the first layer weights

encoding_dim1 = 128
epoch1 = 25

input_img = Input(shape=(input_dim,))
encoded1 = Dense(encoding_dim1, activation='relu')(input_img)
decoded1 = Dense(input_dim, activation='sigmoid')(encoded1)

autoencoder1 = Model(input_img, decoded1)
autoencoder1.compile(optimizer=RMSprop(), loss='binary_crossentropy')
encoder1 = Model(input_img, encoded1)
encoder1.compile(optimizer=RMSprop(), loss='binary_crossentropy')

autoencoder1 = keras.models.load_model('models/autoencoder1.h5')
encoder1 = keras.models.load_model('models/encoder1.h5')
# autoencoder1.fit(x_train
#                  , x_train
#                  , epochs=epoch1
#                  , batch_size=batch_size
#                  , shuffle=False
#                  # , verbose=False
#                  # , validation_data=(x_val, x_val)
#                  , validation_split=0.1
#                  # , callbacks=[early_stopping]
#                  )
# autoencoder1.save('models/autoencoder1.h5')
# encoder1.save('models/encoder1.h5')


# ==========================================================
# train the second layer weights

first_layer_code = encoder1.predict(x_train)

encoding_dim2 = 64
epoch2 = 25

encoded_2_input = Input(shape=(encoding_dim1,))
encoded2 = Dense(encoding_dim2, activation='relu')(encoded_2_input)
decoded2 = Dense(encoding_dim1, activation='sigmoid')(encoded2)

autoencoder2 = Model(encoded_2_input, decoded2)
autoencoder2.compile(optimizer=RMSprop(), loss='binary_crossentropy')
encoder2 = Model(encoded_2_input, encoded2)
encoder2.compile(optimizer=RMSprop(), loss='binary_crossentropy')

# pre_train_64 = keras.models.load_model('models/pre_train_64.h5')
autoencoder2.fit(first_layer_code
                 , first_layer_code
                 # , epochs=epoch2
                 , batch_size=batch_size
                 , shuffle=False
                 # , verbose=False
                 # , validation_data=(x_val, x_val)
                 , validation_split=0.1
                 , callbacks=[early_stopping]
                 )
autoencoder2.save('models/pre_train_64.h5')

# todo check why loss is negative at about -39 while training the second autoencoder

