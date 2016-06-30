from util import logger
import os
import time
from StockAI.models import policy
from data import trade
from db.db import Db
import pandas as pd
from StockAI.training import policy_trainer


def predict_today(datatype, timesteps, data_dim=15):
    # log = logger.log
    nowdate = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    x_predict, id_predict, name_predict = trade.get_today(seg_len=timesteps,
                                                          datatype=datatype,
                                                          split=0.1,
                                                          debug=False)
    network = policy.LSTMPolicy.create_network(timesteps=timesteps,
                                               data_dim=data_dim)
    USER_HOME = os.environ['HOME']
    out_directory_path = USER_HOME + '/dw/'
    meta_file = os.path.join(out_directory_path, 'metadata.json')
    weights_path = policy_trainer.get_best_weights(meta_file)
    network.load_weights(weights_path)

    predicts = network.predict(x_predict, batch_size=16)
    v_predicts = pd.DataFrame()
    v_predicts['code'] = id_predict
    v_predicts['name'] = name_predict
    v_predicts['predict'] = predicts
    v_predicts['datain_date'] = nowdate

    db = Db()
    v_predicts = v_predicts.to_dict('records')
    db.insertmany("""INSERT INTO predicts(code,name,predict,datain_date)
        VALUES (%(code)s,%(name)s,%(predict)s,%(datain_date)s)""", v_predicts)

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
    data_dim = 59
    # 7,15,150
    timesteps = 128
    # nb_classes = 5
    datatype = 'lstm'
    policy_trainer.train(timesteps=timesteps,
                         data_dim=data_dim,
                         datatype=datatype,
                         debug=False,
                         nb_epoch=60,
                         predict_days=2,
                         batch_size=16)
    log.info('train spent time : %s', time.clock() - t_time)

    log.info('predict begin')
    p_time = time.clock()
    predict_today(datatype, timesteps,data_dim=data_dim)
    log.info('predict finished,cost time:%s', time.clock() - p_time)
