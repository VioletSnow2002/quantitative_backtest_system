import numpy as np
import pandas as pd
import feather
from datetime import datetime

global daily, item_map
daily = feather.read_dataframe('../data/stk_daily.feather')
item_map = feather.read_dataframe('../data/stk_fin_item_map.feather')


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

def single_daily(stk_id=None, start_date=datetime(2020,1,1), end_date=datetime(2022,12,31), adj=False):
    df = daily.copy()
    if stk_id!=None:
        df = df[df['stk_id']==stk_id]
    if adj:
        df.loc[:,'open'] *= df['cumadj']
        df.loc[:,'high'] *= df['cumadj']
        df.loc[:,'low'] *= df['cumadj']
        df.loc[:,'close'] *= df['cumadj']
        df = df.drop('cumadj', axis=1)
    return df[(df['date']>=start_date) & (df['date']<=end_date)]

def close_daily(start_date=datetime(2020,1,1), end_date=datetime(2022,12,31)):
    df = daily[['stk_id','date','close','cumadj']]
    df.loc[:,'close'] *= df['cumadj']
    df = df.drop('cumadj', axis=1)
    return df[(df['date']>=start_date) & (df['date']<=end_date)]

def index_daily(start_date=datetime(2020,1,1), end_date=datetime(2022,12,31)):
    close = close_daily()
    date = close['date'].unique()
    close = close.set_index('date')
    ret = close.groupby('stk_id')['close'].apply(lambda x: x.pct_change()).to_frame().reset_index()
    ret.columns = ['stk_id','date','ret']
    ret = ret[['date','ret']].dropna()
    factor = (1 + ret.groupby('date').mean()).cumprod()
    baseline = pd.DataFrame([[datetime(2020,1,2),1]], columns=['date','ret'])
    baseline = baseline.set_index('date')
    return pd.concat([baseline, factor]).loc[start_date:end_date]