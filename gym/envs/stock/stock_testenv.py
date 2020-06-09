import numpy as np
import pandas as pd
from .stock_env import StockEnv

import matplotlib.pyplot as plt


# whole_data = train_daily_data+test_daily_data

class StockTestEnv(StockEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, day = 0, money = 10 , scope = 1):
        super().__init__(day, money, scope)
        self.stop_condition = 685

    def _init_data(self, data, account_growth=None):
        self.daily_data = data
        self.account_growth = account_growth
        self.reset()

    def _plot(self):
        plt.plot(self.asset_memory,'r')
        if not self.account_growth is None:
            plt.plot(self.account_growth)
        plt.savefig('./test_{}.png'.format(self.iteration))
        plt.close()
        print("total_reward:{}".format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))- 10000 ))