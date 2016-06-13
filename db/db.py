#!/usr/bin/env python
#-*- coding:utf-8 -*-

import psycopg2
import psycopg2.extras
from util import logger
from conf import db_settings
from sqlalchemy import create_engine
log = logger.log
"""
获取psql cur
"""


class Db:
    def __init__(self):
        self.settings = db_settings.getdb('core')
        self.host = self.settings['host']
        self.dbname = self.settings['db']
        self.user = self.settings['user']
        self.password = self.settings['passwd']

    def _getPGcur(self, dictcur=False):
        try:
            conn = psycopg2.connect("dbname='" + self.dbname + "' user='" +
                                    self.user + "' host='" + self.host +
                                    "' password='" + self.password + "'")
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            if not dictcur:
                cur = conn.cursor()
            return conn, cur
        except Exception, e:
            log.error('数据库连接获取失败,%s', e)

    def _query(self, cur, sql):
        try:
            cur.execute(sql)
            rows = cur.fetchall()
            return rows
        except Exception, e:
            log.error('sql错误:%s,errsql:%s', e, sql)

    def _queryone(self, cur, sql):
        try:
            cur.execute(sql)
            rows = cur.fetchone()
            return rows
        except Exception, e:
            log.error('sql错误:%s,errsql:%s', e, sql)

    def _get_engine(self):
        engine = create_engine('postgresql+psycopg2://' + self.user + ':' +
                               self.password + '@' + self.host + '/' +
                               self.dbname)
        return engine

    def closedbAll(self, **kwargs):
        for key in kwargs:
            try:
                if key in ['sc', 'hc']:
                    kwargs[key].stop()

                else:
                    kwargs[key].close()
            except Exception, e:
                print "连接关闭失败", e
