from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import GRU, LSTM
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, SReLU


class LSTMPolicy:
    @staticmethod
    def create_network(**kwargs):
        defaults = {"timesteps": 128, "data_dim": 15}
        params = defaults
        params.update(**kwargs)

        network = Sequential()
        network.add(LSTM(output_dim=16,
                         activation='sigmoid',
                         inner_activation='hard_sigmoid',
                         input_shape=(params['timesteps'], params['data_dim']
                                      )))
        network.add(Dropout(0.15))
        network.add(Dense(1))
        # network.add(LeakyReLU(alpha=0.5))
        network.add(Activation('relu'))

        network.compile(optimizer='rmsprop',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        return network


class CNNPolicy:
    @staticmethod
    def create_network(**kwargs):
        defaults = {
            "timesteps": 128,
            "data_dim": 14,
            "nb_filter": 8,
            "nb_row": 2,
            "nb_col": 2
        }
        params = defaults
        params.update(**kwargs)

        network = Sequential()

        # VGG-like
        # input: 13*13 images with 3 channels -> (3, 13, 13) tensors.
        # this applies 32 convolution filters of size 2x2 each.
        # nb_filter+nb_row-1<=timesteps+data_dim-1

        network.add(Convolution2D(params['nb_filter'],
                                  params['nb_row'],
                                  params['nb_col'],
                                  border_mode='valid',
                                  input_shape=(1, params['timesteps'], params[
                                      'data_dim'])))
        network.add(Activation('relu'))
        network.add(Convolution2D(nb_filter, nb_row, nb_col))
        network.add(Activation('relu'))
        network.add(MaxPooling2D(pool_size=(1, 1)))
        network.add(Dropout(0.5))

        network.add(Convolution2D(params['nb_filter'] * 2,
                                  params['nb_row'],
                                  params['nb_col'],
                                  border_mode='valid'))
        network.add(Activation('relu'))
        network.add(Convolution2D(nb_filter * 2, nb_row, nb_col))
        network.add(Activation('relu'))
        network.add(MaxPooling2D(pool_size=(1, 1)))
        network.add(Dropout(0.5))

        network.add(Flatten())
        # Note: Keras does automatic shape inference.
        network.add(Dense(nb_filter * 4))
        network.add(Activation('relu'))
        network.add(Dropout(0.25))

        network.add(Dense(1))
        network.add(Activation('sigmoid'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        # network.compile(optimizer='sgd',
        #                 loss='binary_crossentropy',
        #                 metrics=['accuracy'])
        network.compile(optimizer='rmsprop',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        return network


class MIXPolicy:
    @staticmethod
    def create_network(**kwargs):
        defaults = {
            'timesteps': 128,
            'data_dim': 14,
            'nb_filter': 64,
            'filter_length': 3,
            'pool_length': 2
        }
        params = defaults
        params.update(**kwargs)

        network = Sequential()

        network.add(Convolution1D(nb_filter=params['nb_filter'],
                                  filter_length=params['filter_length'],
                                  border_mode='valid',
                                  activation='relu',
                                  subsample_length=1,
                                  input_shape=(params['timesteps'], params[
                                      'data_dim'])))
        network.add(MaxPooling1D(pool_length=params['pool_length']))
        network.add(Dropout(0.5))

        # network.add(Convolution1D(nb_filter=params['nb_filter'],
        #                           filter_length=params['filter_length'],
        #                           border_mode='valid',
        #                           activation='relu',
        #                           subsample_length=1))
        # network.add(MaxPooling1D(pool_length=params['pool_length']))
        # network.add(Dropout(0.5))

        # network.add(Flatten())
        # # Note: Keras does automatic shape inference.
        # network.add(Dense(params['nb_filter'] * 4))
        # network.add(Activation('relu'))
        # network.add(Dropout(0.25))

        network.add(LSTM(64))
        network.add(Dropout(0.15))
        network.add(Dense(1))
        network.add(Activation('sigmoid'))

        network.compile(optimizer='rmsprop',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        return network
