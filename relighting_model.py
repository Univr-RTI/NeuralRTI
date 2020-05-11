# full(encoder-decoder) relighting model
import keras
#keras.backend.set_floatx('float16')

from keras.models import Input, Model
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute, Lambda
from keras.layers import concatenate
def relight_model(l1,compcoeff):
    keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    inputs1 = Input((l1,))
    ld = Input(shape=(2,))
    encoded1 = Dense(l1, activation='elu', name='dense0')(inputs1)
    encoded2 = Dense(l1, activation='elu', name='dense1')(encoded1)
    encoded3 = Dense(l1, activation='elu', name='dense11')(encoded2)

    encoded = Dense(compcoeff, activation='elu', name='dense12')(encoded3)

    encoder = Model(inputs1, encoded)
    encoder.compile(optimizer=keras.optimizers.Adadelta(), loss='mean_squared_error')
    encode_input = Input(shape=(compcoeff,))
    xx = concatenate([encode_input, ld], axis=-1)
    x = Dense(l1, activation='elu', name='dense2')(xx)
    x = Dense(l1, activation='elu', name='dense3')(x)
    x = Dense(l1, activation='elu', name='dense4')(x)
    decoded = Dense(3, name='dense8')(x)
    decoder = Model([encode_input, ld], decoded)
    autoencoder = Model([inputs1, ld], decoder([encoder(inputs1), ld]))
    return encoder, decoder, autoencoder

