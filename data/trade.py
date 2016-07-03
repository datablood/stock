#!/usr/bin/env python
# -*- coding:utf-8 -*-
from db.db import Db
from util import logger
import numpy as np
import time
import six.moves.cPickle as pickle
import pandas as pd


def get_hist_generator(seg_len=3,
                       debug=False,
                       datatype='cnn',
                       datafile=None,
                       predict_days=18,
                       batch_size=16,
                       df=None):
    while True:
        log = logger.log
        X_batch = []
        Y_batch = []
        begin_time = time.clock()
        k = 0
        predict_days = predict_days

        stockcodes = df['code'].unique()
        stockcodes = np.random.permutation(stockcodes)

        for codes in stockcodes:
            temp_df = df[df.code == codes]
            temp_df1 = temp_df.copy(deep=True)
            temp_df1 = temp_df1.sort_values(by='c_yearmonthday', ascending=1)

            tradedaylist = temp_df1['c_yearmonthday'].values
            tradedaylist.sort()
            tradedaylist = tradedaylist[::-1]

            temp_df1 = temp_df1.set_index('c_yearmonthday')
            if len(tradedaylist) < seg_len:
                log.info('not enough trade days ,code is :%s', codes)
                continue

            i = 0
            for day in tradedaylist:
                i += 1
                segdays = tradedaylist[i + predict_days:i + predict_days +
                                       seg_len]
                segbegin = segdays[len(segdays) - 1]
                segend = segdays[0]
                if len(segdays) < seg_len:
                    break
                data = []
                # for segday in segdays:
                data = temp_df1.loc[segbegin:segend, [
                    'open', 'high', 'close', 'low', 'volume', 'price_change',
                    'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10',
                    'v_ma20', 'turnover', 'deltat', 'BIAS_B', 'BIAS_S',
                    'BOLL_B', 'BOLL_S', 'CCI_B', 'CCI_S', 'DMI_B', 'DMI_HL',
                    'DMI_IF1', 'DMI_IF2', 'DMI_MAX1', 'DMI_S', 'KDJ_B',
                    'KDJ_S', 'KD_B', 'KD_S', 'MACD', 'MACD_B', 'MACD_DEA',
                    'MACD_DIFF', 'MACD_EMA_12', 'MACD_EMA_26', 'MACD_EMA_9',
                    'MACD_S', 'MA_B', 'MA_S', 'PSY_B', 'PSY_MYPSY1', 'PSY_S',
                    'ROC_B', 'ROC_S', 'RSI_B', 'RSI_S', 'VR_B', 'VR_IF1',
                    'VR_IF2', 'VR_IF3', 'VR_S', 'XYYH_B', 'XYYH_B1', 'XYYH_B2',
                    'XYYH_B3', 'XYYH_CC', 'XYYH_DD'
                ]]
                data = data.values
                if datatype == 'cnn':
                    data = [data]
                d1 = tradedaylist[i - 1]
                d3 = tradedaylist[i + predict_days - 1]
                data_tag = temp_df[temp_df.c_yearmonthday == d1][
                    ['code', 'name', 'p_change', 'close']]
                data_tag3 = temp_df[temp_df.c_yearmonthday == d3][
                    ['code', 'name', 'p_change', 'close']]
                temp_y = data_tag['close'].values[0]
                temp_y3 = data_tag3['close'].values[0]
                temp_y = (temp_y - temp_y3) / temp_y3
                temp_y = to_cate01(temp_y)
                X_batch.append(data)
                Y_batch.append(temp_y)
                k += 1
                if k % batch_size == 0:
                    yield (np.asarray(X_batch), np.asarray(Y_batch))
                    X_batch = []
                    Y_batch = []


def get_hist_n_batch(seg_len=3,
                     debug=False,
                     datatype='cnn',
                     datafile=None,
                     predict_days=18,
                     batch_size=16,
                     df=None):
    log = logger.log
    k = 0
    n_batch = 0
    predict_days = predict_days
    stockcodes = df['code'].unique()
    for codes in stockcodes:
        temp_df = df[df.code == codes]
        temp_df1 = temp_df.copy(deep=True)
        temp_df1 = temp_df1.sort_values(by='c_yearmonthday', ascending=1)

        tradedaylist = temp_df1['c_yearmonthday'].values
        tradedaylist.sort()
        tradedaylist = tradedaylist[::-1]

        temp_df1 = temp_df1.set_index('c_yearmonthday')
        if len(tradedaylist) < seg_len:
            log.info('not enough trade days ,code is :%s', codes)
            continue

        i = 0
        for day in tradedaylist:
            i += 1
            segdays = tradedaylist[i + predict_days:i + predict_days + seg_len]
            segbegin = segdays[len(segdays) - 1]
            segend = segdays[0]
            if len(segdays) < seg_len:
                break
            k += 1
            if k % batch_size == 0:
                n_batch += 1
    return n_batch


def get_hist_orgindata(debug=False):
    db = Db()
    engine = db._get_engine()
    sql_stocklist = "select  * from trade_hist where code in (select code  from trade_hist  where high<>0.0 and low <>0.0 group by code having count(code)>100)"
    if debug:
        sql_stocklist += " and code in ('002717','601888','002405')"
    df = pd.read_sql_query(sql_stocklist, engine)
    codes = df['code'].unique()
    # 增加技术指标
    df = add_volatility(df)
    df = get_technique(df)
    return df, codes


def get_predict(debug=False):
    db = Db()
    engine = db._get_engine()
    sql_stocklist = "select * from predicts where datain_date in(select  max(datain_date) from predicts) order by predict desc"
    if debug:
        pass
    df = pd.read_sql_query(sql_stocklist, engine)

    headpredict = df.head(2)
    psummery = df.describe().T
    psummery.columns = ['p_cnt','p_mean','p_std','p_min','p25','p50','p75','p_max']
    return psummery, headpredict


def get_predict_acc1(debug=False):
    db = Db()
    engine = db._get_engine()
    sql_tradehist = "select code,name,p_change from trade_hist where code in (select code from predict_head where c_yearmonthday in (select max(c_yearmonthday) from predict_head) ) order by c_yearmonthday desc"
    sql_predicthead = "select code,predict from predict_head order by c_yearmonthday desc"
    if debug:
        pass
    df_trade = pd.read_sql_query(sql_tradehist, engine).head(2)
    df_predict = pd.read_sql_query(sql_predicthead, engine).head(2)
    df= pd.merge(df_trade,df_predict,on='code')
    df['acc']=(df.p_change).astype(int)
    return df


def get_predict_acc2(debug=False):
    db = Db()
    engine = db._get_engine()
    sql_stocklist = "select  * from acc1"
    if debug:
        pass
    df = pd.read_sql_query(sql_stocklist, engine)
    acc2 = df.sort_values('c_yearmonthday', ascending=0)
    acc2 = acc2.head(2)
    acc2 = acc2.groupby('c_yearmonthday').sum()

    acc2_final = pd.DataFrame()
    acc2_final['h_p_acc'] = [df['acc'].sum() / float(df['acc'].count())]
    acc2_final['h_p_change'] = [df['p_change'].sum()]
    acc2_final['p_acc']=[acc2['acc'].sum()/2.0]
    acc2_final['p_change']=[acc2['p_change'].sum()]

    return acc2_final


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
    df = add_volatility(df)
    stockcodes = df['code'].unique()
    df = get_technique(df)
    print stockcodes

    X_predict = []
    ID_predict = []
    NAME_predict = []
    log.info('begin generate train data and validate data.')
    k = 0
    for codes in stockcodes:
        temp_df = df[df.code == codes]
        temp_df1 = temp_df.copy(deep=True)
        temp_df1 = temp_df1.sort_values(by='c_yearmonthday', ascending=1)

        tradedaylist = temp_df1['c_yearmonthday'].values
        tradedaylist.sort()
        tradedaylist = tradedaylist[::-1]

        temp_df1 = temp_df1.set_index('c_yearmonthday')
        if len(tradedaylist) < seg_len:
            log.info('not enough trade days ,code is :%s', codes)
            continue

        i = 0
        segdays = tradedaylist[i:i + seg_len]
        segbegin = segdays[len(segdays) - 1]
        segend = segdays[0]
        if len(segdays) < seg_len:
            break
        data = []
        data = temp_df1.loc[segbegin:segend, [
            'open', 'high', 'close', 'low', 'volume', 'price_change',
            'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20',
            'turnover', 'deltat', 'BIAS_B', 'BIAS_S', 'BOLL_B', 'BOLL_S',
            'CCI_B', 'CCI_S', 'DMI_B', 'DMI_HL', 'DMI_IF1', 'DMI_IF2',
            'DMI_MAX1', 'DMI_S', 'KDJ_B', 'KDJ_S', 'KD_B', 'KD_S', 'MACD',
            'MACD_B', 'MACD_DEA', 'MACD_DIFF', 'MACD_EMA_12', 'MACD_EMA_26',
            'MACD_EMA_9', 'MACD_S', 'MA_B', 'MA_S', 'PSY_B', 'PSY_MYPSY1',
            'PSY_S', 'ROC_B', 'ROC_S', 'RSI_B', 'RSI_S', 'VR_B', 'VR_IF1',
            'VR_IF2', 'VR_IF3', 'VR_S', 'XYYH_B', 'XYYH_B1', 'XYYH_B2',
            'XYYH_B3', 'XYYH_CC', 'XYYH_DD'
        ]]
        data = data.values
        if datatype == 'cnn':
            data = [data]
        data_tag = temp_df[temp_df.c_yearmonthday == tradedaylist[0]][
            ['code', 'name', 'p_change']]
        temp_id = data_tag['code'].values[0]
        temp_name = data_tag['name'].values[0]
        X_predict.append(data)
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
    # 增加技术指标
    df = add_volatility(df)
    stockcodes = df['code'].unique()
    df = get_technique(df, stockcodes)

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
        temp_df1 = temp_df.copy(deep=True)
        temp_df1 = temp_df1.sort_values(by='c_yearmonthday', ascending=1)

        tradedaylist = temp_df1['c_yearmonthday'].values
        tradedaylist.sort()
        tradedaylist = tradedaylist[::-1]

        temp_df1 = temp_df1.set_index('c_yearmonthday')
        if len(tradedaylist) < seg_len:
            log.info('not enough trade days ,code is :%s', codes)
            continue

        validdays = np.round(split * len(tradedaylist))
        # validdays = 2

        i = 0
        for day in tradedaylist:
            i += 1
            segdays = tradedaylist[i + predict_days:i + predict_days + seg_len]
            segbegin = segdays[len(segdays) - 1]
            segend = segdays[0]
            if len(segdays) < seg_len:
                break
            data = []
            # for segday in segdays:
            data = temp_df1.loc[segbegin:segend, [
                'open', 'high', 'close', 'low', 'volume', 'price_change',
                'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20',
                'turnover', 'deltat', 'BIAS_B', 'BIAS_S', 'BOLL_B', 'BOLL_S',
                'CCI_B', 'CCI_S', 'DMI_B', 'DMI_HL', 'DMI_IF1', 'DMI_IF2',
                'DMI_MAX1', 'DMI_S', 'KDJ_B', 'KDJ_S', 'KD_B', 'KD_S', 'MACD',
                'MACD_B', 'MACD_DEA', 'MACD_DIFF', 'MACD_EMA_12',
                'MACD_EMA_26', 'MACD_EMA_9', 'MACD_S', 'MA_B', 'MA_S', 'PSY_B',
                'PSY_MYPSY1', 'PSY_S', 'ROC_B', 'ROC_S', 'RSI_B', 'RSI_S',
                'VR_B', 'VR_IF1', 'VR_IF2', 'VR_IF3', 'VR_S', 'XYYH_B',
                'XYYH_B1', 'XYYH_B2', 'XYYH_B3', 'XYYH_CC', 'XYYH_DD'
            ]]
            data = data.values
            if datatype == 'cnn':
                data = [data]
            d1 = tradedaylist[i - 1]
            d3 = tradedaylist[i + predict_days - 1]
            data_tag = temp_df[temp_df.c_yearmonthday == d1][
                ['code', 'name', 'p_change', 'close']]
            data_tag3 = temp_df[temp_df.c_yearmonthday == d3][
                ['code', 'name', 'p_change', 'close']]
            temp_y = data_tag['close'].values[0]
            temp_y3 = data_tag3['close'].values[0]
            temp_y = (temp_y - temp_y3) / temp_y3
            temp_y = to_cate01(temp_y)
            temp_id = data_tag['code'].values[0]
            if (i > 0 and i <= validdays):
                X_valid.append(data)
                ID_valid.append(temp_id)
                Y_valid.append(temp_y)
            else:
                X_train.append(data)
                ID_train.append(temp_id)
                Y_train.append(temp_y)
        k += 1
        samples = 12
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
                     'pb', 'mktcap', 'nmc', 'deltat']]
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


def add_volatility(df):
    u = np.log(df['high'] / df['open'])
    d = np.log(df['low'] / df['open'])
    c = np.log(df['close'] / df['open'])
    deltaT = 0.511 * np.square(u - d) - 0.019 * (c * (u + d) - 2 * u *
                                                 d) - 0.383 * np.square(c)
    df['deltat'] = deltaT
    return df


def get_technique(df):
    temp_data = pd.DataFrame()
    stock_data = df.copy(deep=True)
    codes = stock_data['code'].unique()
    for code in codes:
        single_data = stock_data[stock_data.code == code]
        single_data = single_data.sort_values('c_yearmonthday', ascending=1)
        single_data = add_XYYH(single_data)
        single_data = add_MACD(single_data)
        single_data = add_KDJ(single_data)
        single_data = add_BOLL(single_data)
        single_data = add_RSI(single_data)
        single_data = add_BIAS(single_data)
        single_data = add_CCI(single_data)
        single_data = add_DMI(single_data)
        single_data = add_KD(single_data)
        single_data = add_MA(single_data)
        single_data = add_PSY(single_data)
        single_data = add_VR(single_data)
        single_data = add_ROC(single_data)
        temp_data = temp_data.append(single_data)
        temp_data = temp_data.dropna(axis=1)
    return temp_data


'''
  MTM:=C-REF(C,1);
  DX:=100*EMA(EMA(MTM,6),6)/EMA(EMA(ABS(MTM),6),6);
           买:=IF(LLV(DX,N)=LLV(DX,M) AND COUNT(DX<0,2) AND CROSS(DX,MA(DX,2)),1,0);
           买1:=FILTER(买=1,5);
           买2:=DD>CC; DD:=EMA(close,7); CC:=EMA(DD,17);
           买3:=买1 AND 买2;
'''


def add_XYYH(df, N=2, M=7):
    stock_data = df.copy()
    stock_data['XYYH_MTM'] = stock_data['close'].diff()
    stock_data['XYYH_DX'] = 100 * pd.Series.ewm(
        pd.Series.ewm(stock_data['XYYH_MTM'], span=6).mean(),
        span=6).mean() / pd.Series.ewm(
            np.abs(pd.Series.ewm(stock_data['XYYH_MTM'],
                                 span=6).mean()),
            span=6).mean()
    stock_data['XYYH_DX_MA2'] = pd.Series.rolling(stock_data['XYYH_DX'],
                                                  window=2).mean()
    stock_data['XYYH_DX_LLVn'] = pd.Series.rolling(stock_data['XYYH_DX'],
                                                   window=N).min()
    stock_data['XYYH_DX_LLVm'] = pd.Series.rolling(stock_data['XYYH_DX'],
                                                   window=M).min()

    stock_data['XYYH_B'] = 0
    dx_cross = stock_data['XYYH_DX'] > stock_data['XYYH_DX_MA2']
    stock_data.loc[dx_cross[(dx_cross == True) & (dx_cross.shift() == False
                                                  )].index, 'XYYH_B'] = 1
    stock_data.loc[stock_data[stock_data['XYYH_DX_LLVn'] != stock_data[
        'XYYH_DX_LLVm']].index, 'XYYH_B'] = 0
    stock_data.loc[stock_data[(stock_data['XYYH_DX'] >= 0) & (stock_data[
        'XYYH_DX'].shift() >= 0)].index, 'XYYH_B'] = 0

    stock_data['XYYH_B1'] = stock_data['XYYH_B']
    filter_5 = (
        pd.Series.rolling(stock_data['XYYH_B'], window=5).sum() > 1) & (
            stock_data['XYYH_B'] == 1)
    stock_data.loc[filter_5[(filter_5 == True)].index, 'XYYH_B1'] = 0

    stock_data['XYYH_DD'] = pd.Series.ewm(stock_data['close'], span=7).mean()
    stock_data['XYYH_CC'] = pd.Series.ewm(stock_data['XYYH_DD'],
                                          span=17).mean()

    stock_data['XYYH_B2'] = 0
    dd_g_cc = stock_data['XYYH_DD'] > stock_data['XYYH_CC']
    stock_data.loc[dd_g_cc[(dd_g_cc == True)].index, 'XYYH_B2'] = 1

    stock_data['XYYH_B3'] = 0
    buy1_a_buy2 = (stock_data['XYYH_B1'] == 1) & (stock_data['XYYH_B2'] == 1)
    stock_data.loc[buy1_a_buy2[(buy1_a_buy2 == True)].index, 'XYYH_B3'] = 1

    return stock_data


'''
    DIFF:=EMA(close,SHORT) - EMA(close,LONG);
    MACD_DEA  := EMA(DIFF,M);
    MACD := 2*(DIFF-MACD_DEA);
    ENTERLONG:CROSS(MACD,0);
    EXITLONG:CROSS(0,MACD);
'''


def add_MACD(df, SHORT=12, M=9, LONG=26):
    stock_data = df.copy()
    for ma in [SHORT, M, LONG]:
        stock_data['MACD_EMA_' + str(ma)] = pd.Series.ewm(
            stock_data['close'],
            span=ma,
            adjust=True,
            min_periods=0,
            ignore_na=False).mean()

    stock_data['MACD_DIFF'] = stock_data['MACD_EMA_' + str(
        SHORT)] - stock_data[
            'MACD_EMA_' + str(LONG)]
    stock_data['MACD_DEA'] = pd.Series.ewm(stock_data['MACD_DIFF'],
                                           span=M,
                                           adjust=True,
                                           min_periods=0,
                                           ignore_na=False).mean()
    stock_data['MACD'] = 2 * (stock_data['MACD_DIFF'] - stock_data['MACD_DEA'])

    macd_cross1 = stock_data['MACD'] > 0
    macd_cross2 = stock_data['MACD'].shift() < 0
    stock_data['MACD_B'] = 0
    stock_data.loc[stock_data[(macd_cross1 == True) & (macd_cross2 == True
                                                       )].index, 'MACD_B'] = 1

    macd_cross3 = stock_data['MACD'] < 0
    macd_cross4 = stock_data['MACD'].shift() > 0
    stock_data['MACD_S'] = 0
    stock_data.loc[stock_data[(macd_cross3 == True) & (macd_cross4 == True
                                                       )].index, 'MACD_S'] = 1
    return stock_data


'''
    RSV:=(CLOSE-LLV(LOW,N))/(HHV(HIGH,N)-LLV(LOW,N))*100;
    K:=SMA(RSV,M1,1);
    D:=SMA(K,M1,1);
    J:=3*K-2*D;
    ENTERLONG:CROSS(J,0);
    EXITLONG:CROSS(100,J);
'''


def add_KDJ(df, N=9, M1=3.0):
    stock_data = df.copy()
    stock_data['KDJ_LLV_N'] = pd.Series.rolling(stock_data['low'],
                                                window=N).min()
    stock_data['KDJ_HHV_N'] = pd.Series.rolling(stock_data['high'],
                                                window=N).max()
    stock_data['KDJ_RSV'] = (stock_data['close'] - stock_data['KDJ_LLV_N']) / (
        stock_data['KDJ_HHV_N'] - stock_data['KDJ_LLV_N']) * 100
    stock_data['KDJ_K'] = pd.Series.ewm(stock_data['KDJ_RSV'],
                                        adjust=True,
                                        ignore_na=False,
                                        min_periods=0,
                                        alpha=1 / M1).mean()
    stock_data['KDJ_D'] = pd.Series.ewm(stock_data['KDJ_K'],
                                        adjust=True,
                                        ignore_na=False,
                                        min_periods=0,
                                        alpha=1 / M1).mean()
    stock_data['KDJ_J'] = 3 * stock_data['KDJ_K'] - 2 * stock_data['KDJ_D']

    kdj_cross1 = stock_data['KDJ_J'] > 0
    kdj_cross2 = stock_data['KDJ_J'].shift() < 0
    stock_data['KDJ_B'] = 0
    stock_data.loc[stock_data[(kdj_cross1 == True) & (kdj_cross2 == True
                                                      )].index, 'KDJ_B'] = 1

    kdj_cross3 = stock_data['KDJ_J'] < 0
    kdj_cross4 = stock_data['KDJ_J'].shift() > 0
    stock_data['KDJ_S'] = 0
    stock_data.loc[stock_data[(kdj_cross3 == True) & (kdj_cross4 == True
                                                      )].index, 'KDJ_S'] = 1
    return stock_data


''' MID :=MA(CLOSE,N);
    UPPER:=MID+2*STD(CLOSE,N);
    LOWER:=MID-2*STD(CLOSE,N);
    ENTERLONG:CROSS(CLOSE,LOWER);
    EXITLONG:CROSS(CLOSE,UPPER);
'''


def add_BOLL(df, N=20):
    stock_data = df.copy()
    stock_data['BOLL_MID'] = pd.Series.rolling(stock_data['close'],
                                               window=N).mean()
    stock_data['BOLL_UPPER'] = stock_data['BOLL_MID'] + 2 * pd.Series.rolling(
        stock_data['close'], window=N).std()
    stock_data['BOLL_LOWER'] = stock_data['BOLL_MID'] - 2 * pd.Series.rolling(
        stock_data['close'], window=N).std()

    boll_cross1 = stock_data['close'] > stock_data['BOLL_LOWER']
    boll_cross2 = stock_data['close'].shift() < stock_data['BOLL_LOWER']
    stock_data['BOLL_B'] = 0
    stock_data.loc[stock_data[(boll_cross1 == True) & (boll_cross2 == True
                                                       )].index, 'BOLL_B'] = 1

    boll_cross3 = stock_data['close'] < stock_data['BOLL_UPPER']
    boll_cross4 = stock_data['close'].shift() > stock_data['BOLL_UPPER']
    stock_data['BOLL_S'] = 0
    stock_data.loc[stock_data[(boll_cross3 == True) & (boll_cross4 == True
                                                       )].index, 'BOLL_S'] = 1
    return stock_data


''' LC:=REF(CLOSE,1);
    WRSI:=SMA(MAX(CLOSE-LC,0),N,1)/SMA(ABS(CLOSE-LC),N,1)*100;
    ENTERLONG:CROSS(WRSI,LL);
    EXITLONG:CROSS(LH,WRSI);
'''


def add_RSI(df, N=6.0, LL=20, LH=80):
    stock_data = df.copy()
    LC = stock_data['close'].shift()
    MAX = stock_data['close'] - LC
    MAX[MAX < 0] = 0
    ABS = np.abs(stock_data['close'] - LC)
    WRSI = pd.Series.ewm(
        MAX, adjust=True,
        ignore_na=False,
        min_periods=0,
        alpha=1 / N).mean() / pd.Series.ewm(ABS,
                                            adjust=True,
                                            ignore_na=False,
                                            min_periods=0,
                                            alpha=1 / N).mean() * 100

    cross1 = WRSI > LL
    cross2 = WRSI.shift() < LL
    stock_data['RSI_B'] = 0
    stock_data.loc[stock_data[(cross1 == True) & (cross2 == True)].index,
                   'RSI_B'] = 1

    cross3 = LH < WRSI.shift()
    cross4 = LH > WRSI
    stock_data['RSI_S'] = 0
    stock_data.loc[stock_data[(cross3 == True) & (cross4 == True)].index,
                   'RSI_S'] = 1
    return stock_data


''' BIAS:=(CLOSE-MA(CLOSE,N))/MA(CLOSE,N)*100;
    ENTERLONG:CROSS(-LL,BIAS);
    EXITLONG:CROSS(BIAS,LH);'''


def add_BIAS(df, N=12, LL=6, LH=6):
    stock_data = df.copy()
    MA = pd.Series.rolling(stock_data['close'], window=N).mean()
    BIAS = (stock_data['close'] - MA) / MA * 100

    cross1 = BIAS.shift() > -LL
    cross2 = BIAS < -LL
    stock_data['BIAS_B'] = 0
    stock_data.loc[stock_data[(cross1 == True) & (cross2 == True)].index,
                   'BIAS_B'] = 1

    cross3 = LH > BIAS.shift()
    cross4 = LH < BIAS
    stock_data['BIAS_S'] = 0
    stock_data.loc[stock_data[(cross3 == True) & (cross4 == True)].index,
                   'BIAS_S'] = 1
    return stock_data


''' TYP:=(HIGH+LOW+CLOSE)/3;
    CCI:(TYP-MA(TYP,N))/(0.015*AVEDEV(TYP,N));
    INDEX:=CCI(N);
    ENTERLONG:CROSS(INDEX,-100);
    EXITLONG:CROSS(100,INDEX);'''


def add_CCI(df, N=14):
    stock_data = df.copy()
    TYP = (stock_data['high'] + stock_data['low'] + stock_data['close']) / 3
    MA = pd.Series.rolling(TYP, window=N).mean()
    AVEDEV = (TYP - MA).abs().mean()
    CCI = (TYP - MA) / (0.015 * AVEDEV)

    cross1 = CCI.shift() < -100
    cross2 = CCI < -100
    stock_data['CCI_B'] = 0
    stock_data.loc[stock_data[(cross1 == True) & (cross2 == True)].index,
                   'CCI_B'] = 1

    cross3 = CCI.shift() > 100
    cross4 = CCI < 100
    stock_data['CCI_S'] = 0
    stock_data.loc[stock_data[(cross3 == True) & (cross4 == True)].index,
                   'CCI_S'] = 1
    return stock_data


''' MTR:=SUM(MAX(MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1))),ABS(LOW-REF(CLOSE,1))),N);
    HD :=HIGH-REF(HIGH,1);
    LD :=REF(LOW,1)-LOW;
    PDM:=SUM(IF(HD>0&&HD>LD,HD,0),N);
    MDM:=SUM(IF(LD>0&&LD>HD,LD,0),N);
    PDI:=PDM*100/MTR;
    MDI:=MDM*100/MTR;
    ENTERLONG:CROSS(PDI,MDI);
    EXITLONG:CROSS(MDI,PDI);'''


def add_DMI(df, N=14):
    stock_data = df.copy()
    stock_data['DMI_ABSH'] = np.abs(stock_data['high'] - stock_data[
        'close'].shift())
    stock_data['DMI_ABSL'] = np.abs(stock_data['low'] - stock_data[
        'close'].shift())
    stock_data['DMI_HL'] = stock_data['high'] - stock_data['low']
    stock_data['DMI_MAX1'] = stock_data[['DMI_ABSH', 'DMI_HL']].max(axis=1)
    MAX2 = stock_data[['DMI_ABSL', 'DMI_MAX1']].max(axis=1)

    MTR = pd.Series.rolling(MAX2, window=N).sum()
    HD = stock_data['high'].diff()
    LD = stock_data['low'].shift() - stock_data['low']

    stock_data['DMI_IF1'] = 0
    stock_data.loc[stock_data[(HD > 0) & (HD > LD)].index, 'DMI_IF1'] = HD
    stock_data['DMI_IF2'] = 0
    stock_data.loc[stock_data[(LD > 0) & (LD > HD)].index, 'DMI_IF2'] = LD
    PDM = pd.Series.rolling(stock_data['DMI_IF1'], window=N).sum()
    MDM = pd.Series.rolling(stock_data['DMI_IF2'], window=N).sum()
    PDI = PDM * 100 / MTR
    MDI = MDM * 100 / MTR

    cross1 = PDI.shift() < MDI
    cross2 = PDI > MDI
    stock_data['DMI_B'] = 0
    stock_data.loc[stock_data[(cross1 == True) & (cross2 == True)].index,
                   'DMI_B'] = 1

    cross3 = MDI.shift() < PDI
    cross4 = MDI > PDI
    stock_data['DMI_S'] = 0
    stock_data.loc[stock_data[(cross3 == True) & (cross4 == True)].index,
                   'DMI_S'] = 1
    return stock_data


''' WRSV:=(CLOSE-LLV(LOW,N))/(HHV(HIGH,N)-LLV(LOW,N))*100;
    WK:=SMA(WRSV,M1,1);
    D:=SMA(WK,M2,1);
    ENTERLONG:CROSS(WK,D)&&WK<20;
    EXITLONG:CROSS(D,WK)&&WK>80;'''


def add_KD(df, N=9, M1=3.0, M2=3.0):
    stock_data = df.copy()
    stock_data['KD_LLV_N'] = pd.Series.rolling(stock_data['low'],
                                               window=N).min()
    stock_data['KD_HHV_N'] = pd.Series.rolling(stock_data['high'],
                                               window=N).max()
    stock_data['KD_WRSV'] = (stock_data['close'] - stock_data['KD_LLV_N']) / (
        stock_data['KD_HHV_N'] - stock_data['KD_LLV_N']) * 100
    WK = pd.Series.ewm(stock_data['KD_WRSV'],
                       adjust=True,
                       ignore_na=False,
                       min_periods=0,
                       alpha=1 / M1).mean()
    D = pd.Series.ewm(WK,
                      adjust=True,
                      ignore_na=False,
                      min_periods=0,
                      alpha=1 / M2).mean()

    cross1 = (WK.shift() < D.shift()) & (WK > D) & (WK < 20)
    stock_data['KD_B'] = 0
    stock_data.loc[stock_data[(cross1 == True)].index, 'KD_B'] = 1

    cross2 = (D.shift() < WK.shift()) & (WK < D) & (WK > 80)
    stock_data['KD_S'] = 0
    stock_data.loc[stock_data[(cross2 == True)].index, 'KD_S'] = 1
    return stock_data


''' ENTERLONG:CROSS(MA(CLOSE,SHORT),MA(CLOSE,LONG));
    EXITLONG:CROSS(MA(CLOSE,LONG),MA(CLOSE,SHORT));'''


def add_MA(df, S=5, L=20):
    stock_data = df.copy()
    SHORT = pd.Series.rolling(stock_data['low'], window=S).mean()
    LONG = pd.Series.rolling(stock_data['high'], window=L).mean()

    cross1 = (SHORT.shift() < LONG.shift()) & (SHORT > LONG)
    stock_data['MA_B'] = 0
    stock_data.loc[stock_data[(cross1 == True)].index, 'MA_B'] = 1

    cross2 = (SHORT.shift() > LONG.shift()) & (SHORT < LONG)
    stock_data['MA_S'] = 0
    stock_data.loc[stock_data[(cross2 == True)].index, 'MA_S'] = 1
    return stock_data


''' PSY:COUNT(CLOSE>REF(CLOSE,1),N)/N*100;
    PSYMA:MA(PSY,M);
    MYPSY:=PSY(N,1);
    ENTERLONG:CROSS(LL,MYPSY);
    EXITLONG:CROSS(MYPSY,LH);'''


def add_PSY(df, N=12, LL=10, LH=85):
    stock_data = df.copy()
    stock_data['PSY_MYPSY1'] = 0
    stock_data.loc[stock_data[(stock_data['close'].diff() > 0)].index,
                   'PSY_MYPSY1'] = 1
    MYPSY = pd.Series.rolling(stock_data['PSY_MYPSY1'],
                              window=N).sum() / N * 100

    cross1 = (LL < MYPSY.shift()) & (LL > MYPSY)
    stock_data['PSY_B'] = 0
    stock_data.loc[stock_data[(cross1 == True)].index, 'PSY_B'] = 1

    cross2 = (MYPSY.shift() < LH) & (MYPSY > LH)
    stock_data['PSY_S'] = 0
    stock_data.loc[stock_data[(cross2 == True)].index, 'PSY_S'] = 1
    return stock_data


''' ROC:100*(CLOSE-REF(CLOSE,N))/REF(CLOSE,N);
    MAROC:MA(ROC,M);
    WROC:=ROC(N,M);
    ENTERLONG:CROSS(WROC,0);
    EXITLONG:CROSS(0,WROC);'''


def add_ROC(df, N=12, M=6):
    stock_data = df.copy()
    ROC = (stock_data['close'] -
           stock_data['close'].shift(N)) * 100 / stock_data['close'].shift(N)

    cross1 = (ROC.shift() < 0) & (ROC > 0)
    stock_data['ROC_B'] = 0
    stock_data.loc[stock_data[(cross1 == True)].index, 'ROC_B'] = 1

    cross2 = (ROC.shift() > 0) & (ROC < 0)
    stock_data['ROC_S'] = 0
    stock_data.loc[stock_data[(cross2 == True)].index, 'ROC_S'] = 1
    return stock_data


''' WVR := SUM((IF(CLOSE>OPEN,VOL,0)+IF(CLOSE=OPEN,VOL/2,0)),N)/
    SUM((IF(CLOSE<OPEN,VOL,0)+IF(CLOSE=OPEN,VOL/2,0)),N)*100;
    ENTERLONG:CROSS(LL,WVR);
    EXITLONG:CROSS(WVR,LH);'''


def add_VR(df, N=26, LL=70, LH=250):
    stock_data = df.copy()
    CLOSE, OPEN = stock_data['close'], stock_data['open']
    stock_data['VR_IF1'] = stock_data['volume']
    stock_data.loc[stock_data[((CLOSE > OPEN) == False)].index, 'VR_IF1'] = 0
    stock_data['VR_IF2'] = stock_data['volume'] / 2
    stock_data.loc[stock_data[((CLOSE == OPEN) == False)].index, 'VR_IF2'] = 0
    stock_data['VR_IF3'] = stock_data['volume']
    stock_data.loc[stock_data[((CLOSE < OPEN) == False)].index, 'VR_IF3'] = 0

    WVR = pd.Series.rolling(stock_data['VR_IF1'] + stock_data['VR_IF2'],
                            window=N).sum() * 100 / pd.Series.rolling(
                                stock_data['VR_IF2'] + stock_data['VR_IF3'],
                                window=N).sum()

    cross1 = (WVR.shift() > LL) & (WVR < LL)
    stock_data['VR_B'] = 0
    stock_data.loc[stock_data[(cross1 == True)].index, 'VR_B'] = 1

    cross2 = (WVR.shift() < LH) & (WVR > LH)
    stock_data['VR_S'] = 0
    stock_data.loc[stock_data[(cross2 == True)].index, 'VR_S'] = 1
    return stock_data


def add_TRIX(df):
    pass


def add_WR(df):
    pass


def add_ASI(df):
    pass
