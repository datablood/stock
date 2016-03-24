#!/usr/bin/env python
#-*- coding:utf-8 -*-

import psycopg2

"""
获取hivecontext
"""

class Db:

    def _getPGcur(self):
        try:
            conn = psycopg2.connect("dbname='datablood' user='brook' host='127.0.0.1' password='1'")
            #cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur = conn.cursor()
            return conn,cur
        except Exception, e:
            print "数据库连接获取失败", e




    def closedbAll(self,**kwargs):
        for key in kwargs:
            try:
                if key in ['sc','hc']:
                    kwargs[key].stop()

                else:
                    kwargs[key].close()
            except Exception ,e:
                print "连接关闭失败", e


