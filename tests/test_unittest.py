'''
UnitTest for API
@author: brook
'''
import unittest
from stock import trading as td
from stock import newsevent as nv
from stock import billboard as bb


class TestTrading(unittest.TestCase):

    def set_data(self):
        self.code = '600699'
        self.start = '2016-03-22'
        self.end = '2016-03-23'

    def test_tickData(self):
        self.set_data()
        print td.get_tick_data(self.code, date=self.start)


    def test_histData(self):
        self.set_data()
        print td.get_hist_data(self.code, start=self.start, end=self.end)

    def test_latest_news(self):
        self.set_data()
        print nv.get_latest_news(top=5, show_content=False)

    def test_toplist(self):
        self.set_data()
        print bb.top_list()

    def test_notices(self):
        self.set_data()
        print nv.get_notices(self.code)
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
