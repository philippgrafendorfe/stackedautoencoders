from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
# from __future__ import print_function
from IPython.display import SVG, Image
from keras import regularizers, Model
from matplotlib import rc

import keras
import matplotlib.pyplot as plt
import numpy as np

plt.show(block=True)

num_classes = 10
input_dim = 784
batch_size = 256

(x_train, y_train), (x_val, y_val) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.

# x_train = np.concatenate((x_train, x_val))

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))
# print(x_train.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

early_stopping = EarlyStopping(monitor='val_loss', patience=4)

# ======================================================================================================================
# train the first layer weights

encoding_dim1 = 128
epoch1 = 8

input_img = Input(shape=(input_dim,))
encoded1 = Dense(encoding_dim1, activation='relu')(input_img)
# encoded1_bn = BatchNormalization()(encoded1)
decoded1 = Dense(input_dim, activation='relu')(encoded1)
class1 = Dense(num_classes, activation='softmax')(decoded1)

autoencoder1 = Model(input_img, class1)
autoencoder1.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])
encoder1 = Model(input_img, encoded1)
encoder1.compile(optimizer=RMSprop(), loss='binary_crossentropy')

# autoencoder1 = keras.models.load_model('models/autoencoder1.h5')
# encoder1 = keras.models.load_model('models/encoder1.h5')
autoencoder1.fit(x_train
                 , y_train
                 , epochs=epoch1
                 , batch_size=batch_size
                 , shuffle=True
                 # , verbose=False
                 # , validation_data=(x_val, x_val)
                 , validation_split=0.1
                 # , callbacks=[early_stopping]
                 )
# autoencoder1.save('models/autoencoder1.h5')
# encoder1.save('models/encoder1.h5')

score1 = autoencoder1.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score1[0])
print('Test accuracy:', score1[1])

# ==========================================================
# train the second layer weights

first_layer_code = encoder1.predict(x_train)

encoding_dim2 = 64
epoch2 = 5

encoded_2_input = Input(shape=(encoding_dim1,))
encoded2 = Dense(encoding_dim2, activation='relu')(encoded_2_input)
# encoded2_bn = BatchNormalization()(encoded2)
decoded2 = Dense(encoding_dim1, activation='relu')(encoded2)
class2 = Dense(num_classes, activation='softmax')(decoded2)

autoencoder2 = Model(encoded_2_input, class2)
autoencoder2.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])
encoder2 = Model(encoded_2_input, encoded2)
encoder2.compile(optimizer=RMSprop(), loss='binary_crossentropy')

# autoencoder2 = keras.models.load_model('models/autoencoder2.h5')
# encoder2 = keras.models.load_model('models/encoder2.h5')
autoencoder2.fit(first_layer_code
                 , y_train
                 , epochs=epoch2
                 , batch_size=batch_size
                 , shuffle=True
                 # , verbose=False
                 # , validation_data=(x_val, x_val)
                 , validation_split=0.1
                 # , callbacks=[early_stopping]
                 )
# autoencoder2.save('models/autoencoder2.h5')
# encoder2.save('models/encoder2.h5')

first_layer_code_val = encoder1.predict(x_val)

score2 = autoencoder2.evaluate(first_layer_code_val, y_val, verbose=0)
print('Test loss:', score2[0])
print('Test accuracy:', score2[1])

# ==========================================================
# train the third layer weights

second_layer_code = encoder2.predict(encoder1.predict(x_train))

encoding_dim3 = 32
epoch3 = 5

encoded_3_input = Input(shape=(encoding_dim2,))
encoded3 = Dense(encoding_dim3, activation='relu')(encoded_3_input)
# encoded3_bn = BatchNormalization()(encoded3)
decoded3 = Dense(encoding_dim1, activation='relu')(encoded3)
class3 = Dense(num_classes, activation='softmax')(decoded3)

autoencoder3 = Model(encoded_3_input, class3)
autoencoder3.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])
encoder3 = Model(encoded_3_input, encoded3)
encoder3.compile(optimizer=RMSprop(), loss='binary_crossentropy')

# autoencoder2 = keras.models.load_model('models/autoencoder2.h5')
# encoder2 = keras.models.load_model('models/encoder2.h5')
autoencoder3.fit(second_layer_code
                 , y_train
                 , epochs=epoch3
                 , batch_size=batch_size
                 , shuffle=True
                 # , verbose=False
                 # , validation_data=(x_val, x_val)
                 , validation_split=0.1
                 # , callbacks=[early_stopping]
                 )
# autoencoder2.save('models/autoencoder2.h5')
# encoder2.save('models/encoder2.h5')

second_layer_code_val = encoder2.predict(encoder1.predict(x_val))

score3 = autoencoder3.evaluate(second_layer_code_val, y_val, verbose=0)
print('Test loss:', score3[0])
print('Test accuracy:', score3[1])

# ==========================================================
# train the sae

epoch4 = 10

sae_encoded1 = Dense(encoding_dim1, activation='relu')(input_img)
sae_encoded2 = Dense(encoding_dim2, activation='relu')(sae_encoded1)
sae_encoded3 = Dense(encoding_dim3, activation='relu')(sae_encoded2)
sae_decoded1 = Dense(encoding_dim2, activation='relu')(sae_encoded3)
sae_decoded2 = Dense(encoding_dim1, activation='relu')(sae_decoded1)
sae_decoded3 = Dense(input_dim, activation='sigmoid')(sae_decoded2)

sae = Model(input_img, sae_decoded3)

sae.layers[1].set_weights(autoencoder1.layers[1].get_weights())
sae.layers[2].set_weights(autoencoder2.layers[1].get_weights())
sae.layers[3].set_weights(autoencoder3.layers[1].get_weights())
# sae.layers[4].set_weights(autoencoder3.layers[2].get_weights())
# sae.layers[5].set_weights(autoencoder2.layers[2].get_weights())
# sae.layers[6].set_weights(autoencoder1.layers[2].get_weights())

sae.compile(loss='binary_crossentropy', optimizer=RMSprop())
sae.fit(x_train
        , x_train
        , epochs=epoch4
        , batch_size=batch_size
        , shuffle=True
        # , verbose=False
        # , validation_data=(x_val, x_val)
        , validation_split=0.1
        # , callbacks=[early_stopping]
        )

score4 = sae.evaluate(x_val, x_val, verbose=0)
print('Test loss:', score4)

decoded_imgs = sae.predict(x_val)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_val[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


class_layer = Dense(num_classes, activation='softmax')(sae_decoded3)
classifier = Model(inputs=input_img, outputs=class_layer)
classifier.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

classifier.fit(x_train, y_train
               , epochs=7
               , batch_size=batch_size
               , validation_split=0.1
               , shuffle=True)

score5 = classifier.evaluate(x_val, y_val)
print('Test loss:', score5[0])
print('Test accuracy:', score5[1])
