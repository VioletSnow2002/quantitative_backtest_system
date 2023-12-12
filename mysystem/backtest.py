import numpy as np
import pandas as pd
import feather
from datetime import datetime, timedelta

from .signalgen import *

def compute_values(nstocks, trimsize=0, start_date='2020-01-03', end_date='2022-12-31', period=1, rever=True):
    """
    计算反转 (动量) 策略的净值曲线
    """
    close = close_daily()

    portfolio = select_stocks(nstocks=nstocks, trimsize=trimsize, start_date=start_date, end_date=end_date, period=period, rever=rever)
    dates = portfolio.index
    value = pd.Series(1, index=dates)
    nday = len(dates)
    for i in range(nday-1):
        ret = 0
        df = close[close['stk_id'].isin(portfolio.values[i]) & close['date'].isin([dates[i],dates[i+1]])].drop('date', axis=1)
        ret = df.groupby('stk_id').pct_change().dropna().mean() + 1
        value[i+1] = ret
    return value.cumprod()