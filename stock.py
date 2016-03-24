# -*- coding:utf-8 -*-

import sys
import time
reload(sys)
sys.setdefaultencoding('utf-8')
from stock import trading as td
from db.db import Db
from util import dateu as du
from util import logger

'''
stock data
@author: brook
'''

class Stock():

    def set_data(self):
        self.code = '600699'
        self.start = '2016-03-22'
        self.end = '2016-03-23'
        self.nowtime =time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        self.nowdate =time.strftime("%Y-%m-%d",time.localtime(time.time()))


    def getNowdate(self):
        self.set_data()
        return(self.nowdate)

    def test_tickData(self):
        self.set_data()
        print td.get_tick_data(self.code, date=self.start)


    def get_today_all(self):
        self.set_data()
        return(td.get_today_all())

    def insert_today_trade(self):
        self.set_data()
        db = Db()

        conn,cur = db._getPGcur()
        gta = td.get_today_all()
        gta['datain_date']=self.nowtime
        gta['c_yearmonthday']=self.nowdate

        gta=gta.to_dict('records')

        cur.executemany("""INSERT INTO trade_record(c_yearmonthday,code,name,changepercent,trade,open,high,low,settlement
        ,volume,turnoverratio,amount,per,pb,mktcap,nmc,datain_date)
        VALUES (%(c_yearmonthday)s,%(code)s,%(name)s,%(changepercent)s,%(trade)s,%(open)s,%(high)s,%(low)s,%(settlement)s,%(volume)s,%(turnoverratio)s,%(amount)s,%(per)s,%(pb)s,%(mktcap)s,%(nmc)s,%(datain_date)s)""", gta)
        conn.commit()
        cur.close()
        conn.close()



if __name__ == "__main__":
    stock = Stock()
    logger.install({
        'root': {
            'filename': {'DEBUG':"log/debug.log", 'ERROR':'log/err.log'},
        },
    })
    log=logger.log
    try:
        if(not du.is_holiday(stock.getNowdate())):
            log.info('今天是工作日正在同步股票交易数据 .....')
            stock.insert_today_trade()
            log.info('同步完成')
        else:
            log.info('今天是假期，不进行同步数据')
    except Exception ,e:
        log.error('同步数据失败:%s',e)


