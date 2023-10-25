import pathlib
import pandas as pd
import typing
import statsmodels as sm

from pathlib import Path
datalib = Path.cwd().parent.joinpath('data')

sp500 = pd.read_parquet('C:\DESKTOP\PythonEconRepoOnDesktop\ECOPY_23241\data\sp500.parquet', engine='fastparquet')

ff_factors = pd.read_parquet(r'C:\DESKTOP\PythonEconRepoOnDesktop\ECOPY_23241\data\ff_factors.parquet', engine='fastparquet')

df = sp500.merge(ff_factors, left_on='Date', right_on='Date')

df['Excess_Return'] = df['Monthly Returns']-df['RF']

df.sort_values(by='Date', inplace=True)
df['ex_ret_1'] = df.groupby('Symbol')['Excess_Return'].shift(-1)

df.dropna(subset=['ex_ret_1'], inplace=True)
df.dropna(subset=['HML'], inplace=True)

amzn_df = df[df.Symbol == 'AMZN']
amzn_df.drop(columns=["Symbol"], inplace=True)

