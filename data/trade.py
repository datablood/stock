#!/usr/bin/env python
# -*- coding:utf-8 -*-
from db.db import Db
from util import logger
import numpy as np
import time
import six.moves.cPickle as pickle
import pandas as pd


def get_today(split=0.2,
              seg_len=3,
              debug=False,
              datatype='cnn',
              datafile=None):
    log = logger.log
    db = Db()
    engine = db._get_engine()
    sql_stocklist = "select  * from trade_hist where code in (select code  from trade_hist  where high<>0.0 and low <>0.0 group by code having count(code)>100)"
    if debug:
        sql_stocklist += " and code in ('002717','601888','002405')"
    df = pd.read_sql_query(sql_stocklist, engine)
    stockcodes = df['code'].unique()
    print stockcodes

    X_predict = []
    ID_predict = []
    NAME_predict = []
    log.info('begin generate train data and validate data.')
    k = 0
    for codes in stockcodes:
        temp_df = df[df.code == codes]

        tradedaylist = temp_df.copy(deep=True)['c_yearmonthday'].values
        tradedaylist.sort()
        tradedaylist = tradedaylist[::-1]
        if len(tradedaylist) < seg_len:
            log.info('not enough trade days ,code is :%s', codes)
            continue

        i = 0
        segdays = tradedaylist[i:i + seg_len]
        if len(segdays) < seg_len:
            break
        data = []
        SEG_X = []
        for segday in segdays:
            data = temp_df[temp_df.c_yearmonthday == segday][
                ['open', 'high', 'close', 'low', 'volume', 'price_change',
                 'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10',
                 'v_ma20', 'turnover']]
            data = data.values
            SEG_X.append(data[0])
        if datatype == 'cnn':
            SEG_X = [SEG_X]
        data_tag = temp_df[temp_df.c_yearmonthday == tradedaylist[0]][
            ['code', 'name', 'p_change']]
        temp_id = data_tag['code'].values[0]
        temp_name = data_tag['name'].values[0]
        X_predict.append(SEG_X)
        ID_predict.append(temp_id)
        NAME_predict.append(temp_name)
        k += 1
        log.info('%s stock finished ', k)
    return (np.asarray(X_predict), np.asarray(ID_predict),
            np.asarray(NAME_predict))


def get_hist6years(split=0.2,
                   seg_len=3,
                   debug=False,
                   datatype='cnn',
                   datafile=None,
                   predict_days=18):
    log = logger.log
    db = Db()
    engine = db._get_engine()
    sql_stocklist = "select  * from trade_hist where code in (select code  from trade_hist  where high<>0.0 and low <>0.0 group by code having count(code)>100)"
    if debug:
        sql_stocklist += " and code in ('002717','601888','002405')"
    df = pd.read_sql_query(sql_stocklist, engine)
    stockcodes = df['code'].unique()
    print stockcodes

    X_train = []
    X_valid = []
    Y_train = []
    Y_valid = []
    ID_train = []
    ID_valid = []
    log.info('begin generate train data and validate data.')
    begin_time = time.clock()
    k = 0
    predict_days = predict_days
    for codes in stockcodes:
        temp_df = df[df.code == codes]

        tradedaylist = temp_df.copy(deep=True)['c_yearmonthday'].values
        tradedaylist.sort()
        tradedaylist = tradedaylist[::-1]
        if len(tradedaylist) < seg_len:
            log.info('not enough trade days ,code is :%s', codes)
            continue

        validdays = np.round(split * len(tradedaylist))
        # validdays = 2

        i = 0
        for day in tradedaylist:
            i += 1
            segdays = tradedaylist[i + predict_days:i + predict_days + seg_len]
            if len(segdays) < seg_len:
                break
            SEG_X = []
            data = []
            for segday in segdays:
                data = temp_df[temp_df.c_yearmonthday == segday][
                    ['open', 'high', 'close', 'low', 'volume', 'price_change',
                     'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10',
                     'v_ma20', 'turnover']]
                data = data.values
                SEG_X.append(data[0])
            # SEG_X=np.array(SEG_X).T
            if datatype == 'cnn':
                SEG_X = [SEG_X]
            d1 = tradedaylist[i - 1]
            d3 = tradedaylist[i + predict_days - 1]
            # print d1, d3, segdays
            data_tag = temp_df[temp_df.c_yearmonthday == d1][
                ['code', 'name', 'p_change', 'close']]
            data_tag3 = temp_df[temp_df.c_yearmonthday == d3][
                ['code', 'name', 'p_change', 'close']]
            temp_y = data_tag['close'].values[0]
            temp_y3 = data_tag3['close'].values[0]
            temp_y = (temp_y - temp_y3) / temp_y3
            temp_y = to_cate01(temp_y)
            # print 'tempy:', temp_y
            temp_id = data_tag['code'].values[0]
            if (i > 0 and i <= validdays):
                X_valid.append(SEG_X)
                ID_valid.append(temp_id)
                Y_valid.append(temp_y)
            else:
                X_train.append(SEG_X)
                ID_train.append(temp_id)
                Y_train.append(temp_y)
        k += 1
        samples = 10
        if k % samples == 0:
            print k
            log.info('%s stock finished ', k)
            yield ((np.asarray(X_train), np.asarray(Y_train),
                    np.asarray(ID_train)),
                   (np.asarray(X_valid), np.asarray(Y_valid),
                    np.asarray(ID_valid)))
            X_train = []
            X_valid = []
            Y_train = []
            Y_valid = []
            ID_train = []
            ID_valid = []

    yield ((np.asarray(X_train), np.asarray(Y_train), np.asarray(ID_train)),
           (np.asarray(X_valid), np.asarray(Y_valid), np.asarray(ID_valid)))

    # log.info('generate data finished ,cost time:%s', time.clock() - begin_time)
    # log.info('X_train shape is :%s', np.asarray(X_train).shape)
    # log.info('Y_train shape is :%s', np.asarray(Y_train).shape)
    # log.info('X_valid shape is :%s', np.asarray(X_valid).shape)
    # log.info('Y_valid shape is :%s', np.asarray(Y_valid).shape)

    # # X_train=normalize(X_train)
    # # X_valid=normalize(X_valid)

    # if debug:
    #     print(np.asarray(X_train), np.asarray(Y_train),
    #           np.asarray(ID_train)), (np.asarray(X_valid), np.asarray(Y_valid),
    #                                   np.asarray(ID_valid))
    # pickle.dump(
    #     ((np.asarray(X_train), np.asarray(Y_train), np.asarray(ID_train)),
    #      (np.asarray(X_valid), np.asarray(Y_valid), np.asarray(ID_valid))),
    #     open(datafile, 'wb'))


def get_histdata(split=0.15, seg_len=3, debug=False, datatype='cnn'):
    db = Db()
    engine = db._get_engine()
    sql_stocklist = "select  * from trade_record where code in (select code  from trade_record  where high<>0.0 and low <>0.0 group by code having count(code)=(select count(distinct c_yearmonthday) from trade_record))"
    if debug:
        sql_stocklist += " and code in ('300138','002372')"
    df = pd.read_sql_query(sql_stocklist, engine)
    stockcodes = df['code'].unique()

    X_train = []
    X_valid = []
    Y_train = []
    Y_valid = []
    ID_train = []
    ID_valid = []
    log.info('begin generate train data and validate data.')
    begin_time = time.clock()
    k = 0
    for codes in stockcodes:
        temp_df = df[df.code == codes]

        tradedaylist = temp_df.copy(deep=True)['c_yearmonthday'].values
        tradedaylist.sort()
        tradedaylist = tradedaylist[::-1]
        if len(tradedaylist) < seg_len:
            log.info('not enough trade days ,code is :%s', codes)
            continue

        validdays = np.round(split * len(tradedaylist))

        i = 0
        for day in tradedaylist:
            i += 1
            segdays = tradedaylist[i:i + seg_len]
            if len(segdays) < seg_len:
                break
            SEG_X = []
            data = []
            for segday in segdays:
                data = temp_df[temp_df.c_yearmonthday == segday][
                    ['changepercent', 'trade', 'open', 'high', 'low',
                     'settlement', 'volume', 'turnoverratio', 'amount', 'per',
                     'pb', 'mktcap', 'nmc']]
                data = data.values
                SEG_X.append(data[0])
            # SEG_X=np.array(SEG_X).T
            if datatype == 'cnn':
                SEG_X = [SEG_X]
            data_tag = temp_df[temp_df.c_yearmonthday == day][
                ['code', 'name', 'changepercent']]
            temp_y = data_tag['changepercent'].values[0]
            temp_y = to_cate01(temp_y)
            temp_id = data_tag['code'].values[0]
            if (i > 0 and i <= validdays):
                X_valid.append(SEG_X)
                ID_valid.append(temp_id)
                Y_valid.append(temp_y)
            else:
                X_train.append(SEG_X)
                ID_train.append(temp_id)
                Y_train.append(temp_y)
        k += 1
        if k % 500 == 0:
            log.info('%s stock finished ', k)

    log.info('generate data finished ,cost time:%s', time.clock() - begin_time)
    log.info('X_train shape is :%s', np.asarray(X_train).shape)
    log.info('Y_train shape is :%s', np.asarray(Y_train).shape)
    log.info('X_valid shape is :%s', np.asarray(X_valid).shape)
    log.info('Y_valid shape is :%s', np.asarray(Y_valid).shape)

    # X_train=normalize(X_train)
    # X_valid=normalize(X_valid)

    if debug:
        print(np.asarray(X_train), np.asarray(Y_train),
              np.asarray(ID_train)), (np.asarray(X_valid), np.asarray(Y_valid),
                                      np.asarray(ID_valid))
        print(np.asarray(X_train[0][0][0]))
    pickle.dump(
        ((np.asarray(X_train), np.asarray(Y_train), np.asarray(ID_train)),
         (np.asarray(X_valid), np.asarray(Y_valid), np.asarray(ID_valid))),
        open(datatype + '_seg' + str(seg_len) + '.pkl', 'wb'))


def to_catemul(y):
    temp_y = y
    if temp_y < -5:
        temp_y = 0
    elif temp_y >= -5 and temp_y <= -0.2:
        temp_y = 1
    elif temp_y > -0.2 and temp_y < 0.2:
        temp_y = 2
    elif temp_y >= 0.2 and temp_y <= 5:
        temp_y = 3
    else:
        temp_y = 4
    return temp_y


def to_cate01(y):
    temp_y = y
    if temp_y < 0:
        temp_y = 0
    else:
        temp_y = 1
    return temp_y


# sklearn have normalize method
def normalize(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm
