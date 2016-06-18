from data import trade
from StockAI.models import policy
from util import logger
import os
import pandas as pd


def train(timesteps=15, datatype='lstm', debug=False, nb_epoch=50,predict_days=18):
    USER_HOME = os.environ['HOME']
    log = logger.log
    network = policy.LSTMPolicy.create_network(timesteps=timesteps)

    datatype = 'lstm'
    hist6years = trade.get_hist6years(seg_len=timesteps,
                                      datatype=datatype,
                                      split=0.1,
                                      debug=debug,predict_days=predict_days)

    for (x_train, y_train, id_train), (x_valid, y_valid,
                                       id_valid) in hist6years:
        log.info('x_train shape is: %s', x_train.shape)
        # x_train=np_utils.normalize(x_train)
        # x_valid=np_utils.normalize(x_valid)
        # print('y_valid value', type(y_valid[0]), y_valid, type(y_valid))
        # y_train=np_utils.to_categorical(y_train,nb_classes)
        # y_valid=np_utils.to_categorical(y_valid,nb_classes)

        # lstm for a binary classification problem
        weights_path = USER_HOME + '/dw/' + datatype + '_seg' + str(
            timesteps) + '.h5'
        if os.path.exists(weights_path):
            network.load_weights(weights_path)
        # lstm end

        network.fit(x_train,
                    y_train,
                    batch_size=16,
                    nb_epoch=nb_epoch,
                    validation_data=(x_valid, y_valid))
        predicts = network.predict(x_valid, batch_size=16)
        v_predicts = pd.DataFrame()
        v_predicts['CODE'] = id_valid
        v_predicts['PREDICT'] = predicts
        log.info('batch predicts is : %s', v_predicts)

        score = network.evaluate(x_valid, y_valid, verbose=0)
        log.info('Test score:%s', score[0])
        log.info('Test accuracy:%s', score[1])
        network.save_weights(weights_path, overwrite=True)
