import numpy as np
import pandas as pd
import feather
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .backtest import *

def summary(value, rf = 0.02):
    """
    计算策略的年化收益, 年化波动, 夏普比率, 最大回撤, beta, alpha, 超额收益
    """
    dates = value.index
    v = value.values
    n_hold = len(dates) - 1 # 持有期时长
    r = value.values[-1]**(252/n_hold) - 1
    sigma = np.sqrt((value.pct_change().dropna()).var()*252*n_hold/(n_hold-1))
    sharpe = (r-rf) / sigma

    mdd = 0
    for i in range(n_hold):
        for j in range(i, n_hold+1):
            if (v[i]-v[j])/v[i]>mdd:
                mdd = (v[i]-v[j])/v[i]

    idx = index_daily(start_date=dates.min().strftime('%Y-%m-%d'), 
                      end_date=dates.max().strftime('%Y-%m-%d')).ret
    excess = value - idx

    y = value.pct_change().dropna()
    x = idx.pct_change().dropna()
    beta = ((y*x).mean()-y.mean()*x.mean()).sum() / ((x*x).mean()-x.mean()*x.mean()).sum()
    rm = idx.values[-1]**(252/n_hold) - 1
    alpha = r - (rf + beta*(rm-rf))

    return {'annualized rate of return': r, 'annualized volatility': sigma, 
            'sharpe ratio': sharpe, 'maximum drawdown': mdd, 'beta': beta,
            'alpha': alpha, 'excess return': excess}

def show_result(value, rf=0.02):
    """
    展示策略的回测结果
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    dates = value.index
    idx = index_daily(start_date=dates.min().strftime('%Y-%m-%d'), 
                      end_date=dates.max().strftime('%Y-%m-%d')).ret
    idx = idx/idx[0]
    plt.figure(figsize=(10,6))
    plt.plot(dates, value.values-1)
    plt.plot(dates, idx.values-1)
    plt.plot(dates, value.values-idx.values)
    plt.legend(labels=['我的策略','基准收益','超额收益'])
    plt.plot(dates, np.zeros(len(dates)), linestyle='dotted')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    sm = summary(value, rf=rf)
    r, sigma, sharpe, mdd, beta, alpha = list(sm.values())[:-1]
    text = [str(round(100*r,2))+'%',str(round(100*sigma,2))+'%',round(sharpe,4),
            str(round(100*mdd,2))+'%',round(beta,4),round(alpha,4)]
    labels = ['年化收益','年化波动','夏普比率','最大回撤','beta','alpha']
    plt.table(cellText=[text], 
              colLabels=labels, loc='top')