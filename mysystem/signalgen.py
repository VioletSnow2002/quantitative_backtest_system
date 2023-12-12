import numpy as np
import pandas as pd
import feather
from datetime import datetime, timedelta

from .load import *

def signal(start_date='2020-01-03', end_date='2022-12-31', period=1):
    """
    计算特定时间段内所有股票的前 n 日收益率
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    close = close_daily()
    close = close.set_index('date')
    ret = close.groupby('stk_id')['close'].apply(lambda x: x.pct_change(period)).to_frame().reset_index()
    ret.columns = ['stk_id','date','ret']
    ret = ret[(ret['date']>=start_date) & (ret['date']<=end_date)]
    return ret.dropna()

def select_stocks(nstocks, trimsize=0, start_date='2020-01-03', end_date='2022-12-31', period=1, rever=True):
    """
    在特定时间段内生成前 n 日收益率最低 (高) 的nstock只股票
    """
    date = []
    stock = []
    df = signal(start_date=start_date, end_date=end_date, period=period)
    for subdf in df.groupby('date'):
        date.append(subdf[0])
        if rever:
            stock.append(list(subdf[1].sort_values('ret')['stk_id'][trimsize:trimsize+nstocks]))
        elif trimsize!=0:
            stock.append(list(subdf[1].sort_values('ret')['stk_id'][-(trimsize+nstocks):-trimsize]))
        else:
            stock.append(list(subdf[1].sort_values('ret')['stk_id'][-nstocks:]))
    return pd.Series(stock, index=date)