# -*- coding:utf-8 -*-
import json
import urllib,urllib2
import sys
import csv
from os.path import exists
reload(sys)
sys.setdefaultencoding('utf-8')


class Stock:
    sinaapi='http://hq.sinajs.cn/list=sh600677'
    output='json'

    def _getstockdata(self, api=sinaapi, query='', region=''):
        url=api
        try:
            urlopen=urllib2.urlopen(url,timeout=60*60)
            data = urlopen.read()
            data = unicode(data, "gb2312").encode("utf8")
            #js = json.loads(data, 'utf-8')
            data = "{"+data.replace('var ','').replace('\n','').replace(';','').replace('=',':').replace('hq_str_sh600677','\"hq_str_sh600677\"')+"}"
            #cdata = json.loads(json.dumps(data))
            ddata = dict(eval(data))

            print type(ddata)
            print ddata
            print ddata["hq_str_sh600677"]
        except urllib2.HTTPError, e:
            print e.code,url
            return '','','',''


if  __name__=='__main__':
    stock=Stock()
    stock._getstockdata()
