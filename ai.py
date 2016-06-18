from util import logger
import six.moves.cPickle as pickle
import os
import time
from StockAI.models import policy
from data import trade
from db.db import Db
import pandas as pd
from StockAI.training import policy_trainer


def predict_today(datatype, timesteps):
    # log = logger.log
    nowdate = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    x_predict, id_predict, name_predict = trade.get_today(seg_len=timesteps,
                                                          datatype=datatype,
                                                          split=0.1,
                                                          debug=False)
    network = policy.LSTMPolicy.create_network()
    network.load_weights(USER_HOME + '/dw/' + datatype + '_seg' + str(
        timesteps) + '.h5')

    predicts = network.predict(x_predict, batch_size=16)
    v_predicts = pd.DataFrame()
    v_predicts['code'] = id_predict
    v_predicts['name'] = name_predict
    v_predicts['predict'] = predicts
    v_predicts['datain_date'] = nowdate

    db = Db()
    conn, cur = db._getPGcur()
    v_predicts = v_predicts.to_dict('records')
    cur.executemany("""INSERT INTO predicts(code,name,predict,datain_date)
        VALUES (%(code)s,%(name)s,%(predict)s,%(datain_date)s)""", v_predicts)
    conn.commit()
    cur.close()
    conn.close()

    log.info('predicts finished')


if __name__ == "__main__":
    USER_HOME = os.environ['HOME']
    logger.install({
        'root': {
            'filename': {'DEBUG': USER_HOME + "/log/debug.log",
                         'ERROR': USER_HOME + "/log/err.log"},
        },
    })
    log = logger.log
    t_time = time.clock()
    data_dim = 14
    # 7,15,150
    timesteps = 250
    # nb_classes = 5
    datatype = 'lstm'
    policy_trainer.train(timesteps,
                         datatype,
                         debug=False,
                         nb_epoch=200,
                         predict_days=9)
    log.info('train spent time : %s', time.clock() - t_time)

    log.info('predict begin')
    p_time = time.clock()
    predict_today(datatype, timesteps)
    log.info('predict finished,cost time:%s', time.clock() - p_time)
