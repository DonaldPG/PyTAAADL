# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 14:12:38 2019

@author: Don
"""

# --------------------------------------------------
# Build standard keras model
# --------------------------------------------------
def build_model(Xtrain, number_feature_maps, perform_batch_normalization,
                use_leaky_relu, leaky_relu_alpha, use_separable,
                use_dropout, dropout_pct):

    #from keras import backend as K
    #from keras.models import Input
    #from keras.models import Model
    from keras.models import Sequential
    from keras.models import model_from_json
    from keras.layers import Conv2D
    from keras.layers import SeparableConv2D
    from keras.layers import Activation
    from keras.layers import LeakyReLU
    #from keras.layers import MaxPooling2D
    from keras.layers import Dropout
    from keras.layers import Dense
    from keras.layers import Flatten
    #from keras.layers import GaussianNoise
    #from keras.initializers import Initializer
    from keras.layers.normalization import BatchNormalization
    #from keras.callbacks import ModelCheckpoint
    #from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    #from keras.optimizers import RMSprop, Adam, Adagrad, Nadam

    model = Sequential()

    for i, nfeature in enumerate(number_feature_maps):
        if i == 0:
            model.add(Conv2D(nfeature, kernel_size=(3, 1), padding='same',
                             strides=(1, 1), data_format='channels_last',
                             kernel_initializer='lecun_uniform', bias_initializer='zeros',
                             input_shape=(Xtrain.shape[1], Xtrain.shape[2], 3)))
        else:
            if perform_batch_normalization == True:
                model.add(Conv2D(nfeature, kernel_size=(3, 1), padding='same',
                             strides=(1, 1), data_format='channels_last',
                             use_bias=False))
                model.add(BatchNormalization())
            else:
                model.add(Conv2D(nfeature, kernel_size=(3, 1), padding='same',
                             strides=(1, 1), data_format='channels_last',
                             use_bias=True))
        if use_leaky_relu == False:
            model.add(Activation('relu'))
        else:
            model.add(LeakyReLU(alpha=leaky_relu_alpha))
        if use_dropout == True and i > 2:
            model.add(Dropout(dropout_pct))
        #if _is_odd(i) or i == number_feature_maps[-1]:
        #    model.add(MaxPooling2D(pool_size=(2, 1)))
        print("i, model shape = ", i, model.output_shape)

    nfeature = 16
    for i in range(2):
        if use_separable == False:
            model.add(Conv2D(nfeature, kernel_size=(1, 3), padding='same', strides=(1, 1),
                             data_format='channels_last', use_bias=False))
        else:
            model.add(Dense(16))
            print("i, before separableConv2D, model shape = ", i, model.output_shape)
            model.add(SeparableConv2D(nfeature, kernel_size=(1, 3), padding='same',
                                      strides=(1, 1), data_format='channels_last',
                                      depth_multiplier=1, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        if use_dropout == True:
            model.add(Dropout(dropout_pct))

    model.add(Flatten())
    model.add(Dense(int(64./dense_factor), kernel_initializer='lecun_uniform', bias_initializer='zeros'))
    model.add(Dense(int(32./dense_factor)))
    model.add(Dense(int(16./dense_factor)))
    model.add(Dense(1))
    print("final model shape = ", model.output_shape)

    return model

# --------------------------------------------------
# Build keras model with squeeze-excite block
# --------------------------------------------------

def build_se_model(Xtrain, number_feature_maps, perform_batch_normalization,
                   use_leaky_relu, use_separable,
                   leaky_relu_alpha, dense_factor):

    from keras import backend as K
    from keras.models import Input
    from keras.models import Model
    #from keras.models import Sequential
    #from keras.models import model_from_json
    from keras.layers import Conv2D
    from keras.layers import SeparableConv2D
    from keras.layers import Activation
    from keras.layers import LeakyReLU
    #from keras.layers import MaxPooling2D
    #from keras.layers import Dropout
    from keras.layers import Dense
    from keras.layers import Flatten
    #from keras.layers import GaussianNoise
    #from keras.initializers import Initializer
    from keras.layers.normalization import BatchNormalization
    #from keras.callbacks import ModelCheckpoint
    #from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    #from keras.optimizers import RMSprop, Adam, Adagrad, Nadam
    
    from functions.se import squeeze_excite_block

    #model = Sequential()
    K.clear_session()
    inputs = Input(shape=((Xtrain.shape[1], Xtrain.shape[2], 3)))

    for i, nfeature in enumerate(number_feature_maps):
        if i == 0:
            x = Conv2D(nfeature, kernel_size=(3, 1), padding='same',
                             strides=(1, 1), data_format='channels_last',
                             kernel_initializer='lecun_uniform', bias_initializer='zeros')(inputs)
        else:
            if perform_batch_normalization == True:
                x = Conv2D(nfeature, kernel_size=(3, 1), padding='same',
                             strides=(1, 1), data_format='channels_last',
                             use_bias=False)(x)
                x = BatchNormalization()(x)
            else:
                x = Conv2D(nfeature, kernel_size=(3, 1), padding='same',
                             strides=(1, 1), data_format='channels_last',
                             use_bias=True)(x)
        #model.add(GaussianNoise(.0001))
        if use_leaky_relu == False:
            x = Activation('relu')(x)
        else:
            x = LeakyReLU(alpha=leaky_relu_alpha)(x)
        #if i > 2:
        #    x = Dropout(dropout_pct)(x)
        #if _is_odd(i) or i == number_feature_maps[-1]:
        #    model.add(MaxPooling2D(pool_size=(2, 1)))
        print("i, model shape = ", i, K.shape(x))

    x = squeeze_excite_block(x, ratio=16)

    nfeature = 16
    for i in range(2):
        if use_separable == False:
            x = Conv2D(nfeature, kernel_size=(1, 3), padding='same', strides=(1, 1),
                             data_format='channels_last', use_bias=False)(x)
        else:
            x = Dense(16)(x)
            print("i, before separableConv2D, model shape = ", i, K.shape(x))
            x = SeparableConv2D(nfeature, kernel_size=(1, 3), padding='same',
                                      strides=(1, 1), data_format='channels_last',
                                      depth_multiplier=1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(int(64./dense_factor), kernel_initializer='lecun_uniform', bias_initializer='zeros')(x)
    x = Dense(int(32./dense_factor))(x)
    x = Dense(int(16./dense_factor))(x)
    predictions = Dense(1)(x)
    model = Model(inputs=inputs, outputs=predictions)
    print("final model shape = ", model.output_shape)

    return model

