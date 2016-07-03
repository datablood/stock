#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
存储预测及准确率等数据
'''

from db.db import Db
from data import trade
import time

def insert_predict_statics():
    db = Db()
    nowdate = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    psummery,headpredict = trade.get_predict()
    psummery['c_yearmonthday'] = nowdate
    headpredict['c_yearmonthday'] = nowdate
    psummery=psummery.to_dict('records')
    headpredict=headpredict.to_dict('records')
    db.insertmany("""INSERT INTO predict_head(c_yearmonthday,code,name,predict)
        VALUES (%(c_yearmonthday)s,%(code)s,%(name)s,%(predict)s)""", headpredict)
    db.insertmany("""INSERT INTO predict_statics(c_yearmonthday,p_cnt,p_mean,p_std,p_min,p25,p50,p75,p_max)
        VALUES (%(c_yearmonthday)s,%(p_cnt)s,%(p_mean)s,%(p_std)s,%(p_min)s,%(p25)s,%(p50)s,%(p75)s,%(p_max)s)""", psummery)

def insert_predict_acc():
    db = Db()
    nowdate = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    acc1 = trade.get_predict_acc1()
    acc1['c_yearmonthday'] = nowdate
    acc1=acc1.to_dict('records')
    db.insertmany("""INSERT INTO acc1(c_yearmonthday,code,name,predict,p_change,acc)
        VALUES (%(c_yearmonthday)s,%(code)s,%(name)s,%(predict)s,%(p_change)s,%(acc)s)""", acc1)

    acc2 = trade.get_predict_acc2()
    acc2['c_yearmonthday'] = nowdate
    acc2=acc2.to_dict('records')
    db.insertmany("""INSERT INTO acc2(c_yearmonthday,p_acc,p_change,h_p_acc,h_p_change)
        VALUES (%(c_yearmonthday)s,%(p_acc)s,%(p_change)s,%(h_p_acc)s,%(h_p_change)s)""", acc2)



if __name__ == '__main__':
    insert_predict_statics()
    insert_predict_acc()


