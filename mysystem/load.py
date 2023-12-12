import numpy as np
import pandas as pd
import feather
from datetime import datetime, timedelta

global daily_df, item_map, balance_df, income_df, cashflow_df, annotation_df
daily_df = feather.read_dataframe('../data/stk_daily.feather')
item_map = feather.read_dataframe('../data/stk_fin_item_map.feather')
balance_df = feather.read_dataframe('../data/stk_fin_balance.feather')
income_df = feather.read_dataframe('../data/stk_fin_income.feather')
cashflow_df = feather.read_dataframe('../data/stk_fin_cashflow.feather')
annotation_df = feather.read_dataframe('../data/stk_fin_annotation.feather')


def convert(str):
    """
    实现股票财务字段信息映射表中的映射
    """
    try:
        return item_map[item_map['field']==str]['item'].values[0]
    except:
        return str
    
def recover(df):
    """
    根据股票财务字段信息映射表还原各财务数据表中的列名
    """
    df_new = df.copy()
    df_new.columns = list(map(convert, df.columns))
    return df_new

def daily(stk_id=None, start_date='2020-01-01', end_date='2022-12-31', adj=False):
    """
    查询特定时间段内特定股票的日行情
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    df = daily_df.copy()
    if stk_id!=None:
        df = df[df['stk_id']==stk_id]
    if adj:
        df.loc[:,'open'] *= df['cumadj']
        df.loc[:,'high'] *= df['cumadj']
        df.loc[:,'low'] *= df['cumadj']
        df.loc[:,'close'] *= df['cumadj']
        df = df.drop('cumadj', axis=1)
    return df[(df['date']>=start_date) & (df['date']<=end_date)]

def balance(stk_id=None, start_date='1900-01-01', end_date='2022-12-31'):
    """
    查询特定时间段内特定股票的资产负债表
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    df = recover(balance_df)
    if stk_id!=None:
        df = df[df['stk_id']==stk_id]
    return df[(df['date']>=start_date) & (df['date']<=end_date)]

def income(stk_id=None, start_date='1900-01-01', end_date='2022-12-31'):
    """
    查询特定时间段内特定股票的利润表
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    df = recover(income_df)
    if stk_id!=None:
        df = df[df['stk_id']==stk_id]
    return df[(df['date']>=start_date) & (df['date']<=end_date)]

def cashflow(stk_id=None, start_date='1900-01-01', end_date='2022-12-31'):
    """
    查询特定时间段内特定股票的现金流量表
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    df = recover(cashflow_df)
    if stk_id!=None:
        df = df[df['stk_id']==stk_id]
    return df[(df['date']>=start_date) & (df['date']<=end_date)]

def annotation(stk_id=None, start_date='1900-01-01', end_date='2022-12-31'):
    """
    查询特定时间段内特定股票的财务报表附注
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    df = recover(annotation_df)
    if stk_id!=None:
        df = df[df['stk_id']==stk_id]
    return df[(df['date']>=start_date) & (df['date']<=end_date)]

def close_daily(start_date='2020-01-01', end_date='2022-12-31'):
    """
    查询特定时间段内所有股票的复权收盘价
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    df = daily_df[['stk_id','date','close','cumadj']]
    df.loc[:,'close'] *= df['cumadj']
    df = df.drop('cumadj', axis=1)
    return df[(df['date']>=start_date) & (df['date']<=end_date)]

def index_daily(start_date='2020-01-01', end_date='2022-12-31'):
    """
    查询特定时间段内的指数行情
    """
    close = close_daily(start_date=start_date, end_date=end_date)
    close = close.set_index('date')
    ret = close.groupby('stk_id')['close'].apply(lambda x: x.pct_change()).to_frame().reset_index()
    ret.columns = ['stk_id','date','ret']
    ret = ret[['date','ret']].dropna()
    factor = (1 + ret.groupby('date').mean()).cumprod()
    baseline = pd.DataFrame([[datetime.strptime(start_date, '%Y-%m-%d'),1]], columns=['date','ret'])
    baseline = baseline.set_index('date')
    return pd.concat([baseline, factor])