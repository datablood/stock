# -*- coding:utf-8 -*-

from util import logger
from util import dateu as du
import os
import time
from StockAI.models import policy
from data import trade
from db.db import Db
import pandas as pd
from StockAI.training import policy_trainer
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def predict_today(datatype, timesteps, data_dim=15):
    # log = logger.log
    nowdate = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    x_predict, id_predict, name_predict = trade.get_today(seg_len=timesteps,
                                                          datatype=datatype,
                                                          split=0.1,
                                                          debug=False)
    network = policy.MIXPolicy.create_network(timesteps=timesteps,
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
    try:
        if (not du.is_holiday(time.strftime("%Y-%m-%d", time.localtime(time.time())))):
            t_time = time.clock()
            log.info('今天是工作日正在进行训练数据')
            log.info('开始训练数据')
            data_dim = 59
            # 7,15,150
            timesteps = 128
            # nb_classes = 5
            datatype = 'lstm'

            nb_epoch = 28
            if int(time.strftime('%w')) == 5:
                log.info('今天是周五，进行长期训练')
                nb_epoch = 108

            policy_trainer.train(timesteps=timesteps,
                                 data_dim=data_dim,
                                 datatype=datatype,
                                 debug=False,
                                 nb_epoch=nb_epoch,
                                 predict_days=2,
                                 batch_size=16)
            log.info('训练完成,cost time : %s', time.clock() - t_time)

            log.info('正在预测数据')
            p_time = time.clock()
            predict_today(datatype, timesteps, data_dim=data_dim)
            log.info('预测完成,cost time:%s', time.clock() - p_time)
        else:
            log.info('今天是假期，正在进行多次迭代')
    except Exception, e:
        log.error('预测数据失败:%s', e)
