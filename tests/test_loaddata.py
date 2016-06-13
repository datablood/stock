'''
UnitTest for load_data
@author: brook
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from data import trade


class TestLoaddata(unittest.TestCase):
    def test_getData(self):
        trade.get_data()


if __name__ == "__main__":
    unittest.main()
