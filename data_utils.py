import numpy as np
import pandas as pd


def convert_data(data):
    daily_data = []
    for date in np.unique(data.datadate):
        daily_data.append(data[data.datadate == date])
    return daily_data

def load_data():
    data_path = "./gym/envs/stock/Data_Daily_Stock_Dow_Jones_30/dow_jones_30_daily_price.csv"
    # index_path = './gym/envs/zxstock/Data_Daily_Stock_Dow_Jones_30/^DJI.csv'
    # if not index_path is None:
    #     dji = pd.read_csv(index_path)
    #     test_dji=dji[dji['Date']>'2016-01-01']
    #     dji_price=test_dji['Adj Close']
    #     # dji_date = test_dji['Date']
    #     # get percentage change
    #     daily_return = dji_price.pct_change(1)
    #     daily_return=daily_return[1:]
    #     daily_return.reset_index()
    #     initial_amount = 10000
    #     total_amount=initial_amount
    #     account_growth = list()
    #     account_growth.append(initial_amount)
    #     for i in range(len(daily_return)):
    #         total_amount = total_amount * daily_return.iloc[i] + total_amount
    #         account_growth.append(total_amount)

    data_1 = pd.read_csv(data_path)

    equal_4711_list = list(data_1.tic.value_counts() == 4711)
    names = data_1.tic.value_counts().index
    select_stocks_list = list(names[equal_4711_list])+['NKE','KO']
    data_2 = data_1[data_1.tic.isin(select_stocks_list)][~data_1.datadate.isin(['20010912','20010913'])]
    data_3 = data_2[['iid','datadate','tic','prccd','ajexdi']]
    data_3['adjcp'] = data_3['prccd'] / data_3['ajexdi']
    train_data = convert_data(data_3[(data_3.datadate > 20090000) & (data_3.datadate < 20160000)])
    test_data = convert_data(data_3[data_3.datadate > 20160000])
    return train_data, test_data


def save_data(data, cols):
    data = np.array(data)
    df = pd.DataFrame(data, columns=cols)