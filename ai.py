from util import logger
import six.moves.cPickle as pickle
import os
import time

if __name__ == "__main__":
    USER_HOME = os.environ['HOME']
    logger.install({
        'root': {
            'filename': {'DEBUG': USER_HOME + "/log/debug.log",
                         'ERROR': USER_HOME + "/log/err.log"},
        },
    })
    log = logger.log
    s_time=time.clock()

    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Flatten
    from keras.layers import GRU, LSTM
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.optimizers import SGD
    from keras.utils import np_utils
    from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, SReLU
    import numpy as np
    import pandas as pd

    data_dim = 14
    timesteps = 128
    nb_classes = 5
    """
     timesteps :15 split=0.2 loss: 1.2657 - acc: 0.4479 - val_loss: 1.2914 - val_acc: 0.4313
     timesteps :2 split=0.2 loss: 1.1970 - acc: 0.4701 - val_loss: 1.2030 - val_acc: 0.4435
    """

    # expected input data shape: (batch_size, timesteps, data_dim)
    # model = Sequential()
    # model.add(LSTM(32, return_sequences=True,
    #                input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    # model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    # model.add(LSTM(32))  # return a single vector of dimension 32

    # model.add(LSTM(8, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    # model.add(Dense(nb_classes, activation='softmax'))

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])

    datatype = 'lstm'
    # datafile = USER_HOME + '/dw/' + datatype + '_seg' + str(timesteps) + '.pkl'
    from data import trade
    temp_hist = trade.get_hist6years(seg_len=timesteps,
                                     datatype=datatype,
                                     split=0.1,
                                     debug=False)
    # (x_train, y_train, id_train), (
    #     x_valid, y_valid, id_valid) = pickle.load(open(datafile, 'rb'))
    # x_train=np_utils.normalize(x_train)
    # x_valid=np_utils.normalize(x_valid)
    # print('y_valid value', type(y_valid[0]), y_valid, type(y_valid))
    # y_train=np_utils.to_categorical(y_train,nb_classes)
    # y_valid=np_utils.to_categorical(y_valid,nb_classes)
    model = Sequential()
    model.add(LSTM(output_dim=16,
                   activation='sigmoid',
                   inner_activation='hard_sigmoid',
                   input_shape=(timesteps, data_dim)))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    # model.add(LeakyReLU(alpha=0.5))
    model.add(Activation('relu'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    for (x_train, y_train, id_train), (x_valid, y_valid,
                                       id_valid) in temp_hist:
        log.info('x_train shape is: %s', x_train.shape)

        # VGG-like
        # input: 13*13 images with 3 channels -> (3, 13, 13) tensors.
        # this applies 32 convolution filters of size 2x2 each.
        # nb_filter+nb_row-1<=timesteps+data_dim-1
        # nb_filter =8
        # nb_row = 2
        # nb_col = 2

        # model.add(Convolution2D(nb_filter,
        #                         nb_row,
        #                         nb_col,
        #                         border_mode='valid',
        #                         input_shape=(1, timesteps, data_dim)))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(nb_filter, nb_row, nb_col))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(1, 1)))
        # model.add(Dropout(0.5))

        # model.add(Convolution2D(
        #     nb_filter * 2, nb_row,
        #     nb_col, border_mode='valid'))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(nb_filter * 2, nb_row, nb_col))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(1, 1)))
        # model.add(Dropout(0.5))

        # model.add(Flatten())
        # # Note: Keras does automatic shape inference.
        # model.add(Dense(nb_filter * 4))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.25))

        # model.add(Dense(1))
        # model.add(Activation('sigmoid'))

        # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(optimizer='sgd',
        #               loss='binary_crossentropy',
        #               metrics=['accuracy'])
        # vgg end

        # lstm for a binary classification problem
        model.load_weights(USER_HOME + '/dw/' + datatype + '_seg' + str(
            timesteps) + '.h5')
        # lstm end

        model.fit(x_train,
                  y_train,
                  batch_size=16,
                  nb_epoch=20,
                  validation_data=(x_valid, y_valid))
        predicts = model.predict(x_valid, batch_size=16)
        v_predicts=pd.DataFrame()
        v_predicts['CODE']=id_valid
        v_predicts['PREDICT']=predicts
        log.info('batch predicts is : %s', v_predicts)

        score = model.evaluate(x_valid, y_valid, verbose=0)
        log.info('Test score:%s', score[0])
        log.info('Test accuracy:%s', score[1])
        model.save_weights(
            USER_HOME + '/dw/' + datatype + '_seg' + str(timesteps) + '.h5',
            overwrite=True)
    cost_time=time.clock()-s_time
    log.info('train spent time : %s',cost_time)
