# -*- coding:utf-8 -*-

import sys
import time
reload(sys)
sys.setdefaultencoding('utf-8')
from stock import trading as td
from db.db import Db
from util import dateu as du
from util import logger
import os
import pandas as pd
'''
stock data
@author: brook
'''


class Stock():
    def set_data(self):
        self.code = '600677'
        self.name = '航天通信'
        self.nowtime = time.strftime("%Y-%m-%d %H:%M:%S",
                                     time.localtime(time.time()))
        self.nowdate = time.strftime("%Y-%m-%d", time.localtime(time.time()))

    def getNowdate(self):
        self.set_data()
        return (self.nowdate)

    def test_tickData(self):
        self.set_data()
        print td.get_tick_data(self.code, date=self.start)

    def get_today_all(self):
        self.set_data()
        return (td.get_today_all())

    def insert_today_trade(self):
        self.set_data()
        db = Db()

        gta = td.get_today_all()
        gta['datain_date'] = self.nowtime
        gta['c_yearmonthday'] = self.nowdate

        gta = gta.to_dict('records')

        db.insertmany(
            """INSERT INTO trade_record(c_yearmonthday,code,name,changepercent,trade,open,high,low,settlement
        ,volume,turnoverratio,amount,per,pb,mktcap,nmc,datain_date)
        VALUES (%(c_yearmonthday)s,%(code)s,%(name)s,%(changepercent)s,%(trade)s,%(open)s,%(high)s,%(low)s,%(settlement)s,%(volume)s,%(turnoverratio)s,%(amount)s,%(per)s,%(pb)s,%(mktcap)s,%(nmc)s,%(datain_date)s)""",
            gta)

    def insert_hist_trade(self):
        self.set_data()
        db = Db()

        engine = db._get_engine()
        sql_stocklist = "select code,name from stock_code"
        codes = pd.read_sql_query(sql_stocklist, engine)
        codes = codes.to_dict('records')
        i = 1
        for row in codes:
            gta = td.get_hist_data(code=row['code'],
                                   start=self.nowdate,
                                   end=self.nowdate,
                                   ktype='D',
                                   retry_count=3,
                                   pause=0.001)

            gta['datain_date'] = self.nowtime
            gta['code'] = row['code']
            gta['name'] = row['name']
            gta['c_yearmonthday'] = gta.index

            gta = gta.to_dict('records')
            try:
                db.insertmany(
                    """INSERT INTO trade_hist(c_yearmonthday,code,name,open,high,close,low,volume,price_change,p_change,ma5,ma10,ma20,v_ma5,v_ma10,v_ma20,turnover,datain_date)
                VALUES (%(c_yearmonthday)s,%(code)s,%(name)s,%(open)s,%(high)s,%(close)s,%(low)s,%(volume)s,%(price_change)s,%(p_change)s,%(ma5)s,%(ma10)s,%(ma20)s,%(v_ma5)s,%(v_ma10)s,%(v_ma20)s,%(turnover)s,%(datain_date)s)""",
                    gta)
            except Exception, e:
                log.error('insert error:%s ', e)

            log.info('%s stock insert finished,%s,%s', i, row['code'],
                     row['name'].decode('utf-8'))
            i += 1


if __name__ == "__main__":
    stock = Stock()
    USER_HOME = os.environ['HOME']
    logger.install({
        'root': {
            'filename': {'DEBUG': USER_HOME + "/log/debug.log",
                         'ERROR': USER_HOME + "/log/err.log"},
        },
    })

    log = logger.log
    try:
        if (not du.is_holiday(stock.getNowdate())):
            log.info('今天是工作日正在同步股票交易数据 .....')
            log.info('开始同步历史数据')
            stock.insert_today_trade()
            stock.insert_hist_trade()
            log.info('同步完成')
        else:
            log.info('今天是假期，不进行同步数据')
    except Exception, e:
        log.error('同步数据失败:%s', e)
