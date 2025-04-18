# ================================================= Calculate Performance and Risk Measures ============================================================= #

import os
import pandas as pd
import numpy as np

# import a tupling module
from typing import Tuple


# ##### Set the current working directory
# path="e:\\Copy\\SCRIPTS\\Forecast_Stocks\\Code\\"
# os.chdir(path)

def annualized_return(df: pd.DataFrame, ret_column = 'raw_ret'):
    '''Compute Annualized Returns
    INPUT 
        df: a dataframe containing a column of raw returns
    OUTPUT
        a float number of annualized return
    Reference: https://www.investopedia.com/terms/a/annualized-total-return.asp#:~:text=An%20annualized%20total%20return%20is,the%20annual%20return%20was%20compounded.
                        and https://www.assetmacro.com/financial-terms/annualized-return/#:~:text=Annualized%20Return%20is%20the%20average,252%20trading%20days%20of%20year).
    '''
    gross_return = (1. + df[ret_column]).prod()
    days = df.shape[0]

    years = days / 252 # there are about 252 business days per year 
    ann_return = gross_return ** (1/years)
    ann_return = ann_return - 1.
    return ann_return

def annualized_standard_deviation(df: pd.DataFrame, ret_column = 'raw_ret'):
    '''Compute Annualized Standard Deviation
    INPUT 
        df: a dataframe containing a column of raw returns
    OUTPUT
        a float number of annualized standard deviation
    Reference: https://financetrain.com/calculate-annualized-standard-deviation
    '''
    std = df[ret_column].std() * (252 ** 0.5) # there are about 252 business days per year
    return std

def max_drawdown(df: pd.DataFrame, ret_column = 'raw_ret'):
    ''' Change from Max Peak to Trough Loss in a period
    INPUT 
        df: a dataframe containing a column of raw returns
    OUTPUT
        a float number of maximum drawdown '''

    # Calculate cumulative returns
    cum_rets =  (1. + df[ret_column]).cumprod() - 1.

    # Calculate the high water marks (HWM) of cumulative returns
    HWM_cum_rets = cum_rets.cummax()

    # Calculate daily drawdowns
    daily_drawdowns =  HWM_cum_rets - cum_rets

    max_dd = np.nanmax(daily_drawdowns)

    # Plot the results
    # daily_drawdowns.plot()
    
    return max_dd

def gain_to_pain_ratio(df: pd.DataFrame, ret_column = 'raw_ret'):
    ''' Calculate Schwager's Gain to Pain Ratio
    INPUT 
        df: a dataframe containing a column of raw returns
    OUTPUT
        a float number of the gain-to-pain ratio (a Gain to Pain ratio above 1.25 is considered good, and a value over 2 is very good.)
    Reference: https://www.investmacro.com/forex/2018/07/5-statistics-for-analyzing-your-trading-performance 
                        and https://jackschwager.com/market-wizards-search-part-2-the-performance-statistics-i-use '''

    net_return = df[ret_column].sum()
    abs_negative_return = abs( df[ret_column][df[ret_column] < 0].sum() )
    gain_to_pain = net_return / (abs_negative_return + 1e-4)
    return gain_to_pain

def calmar_ratio(df: pd.DataFrame, ret_column = 'raw_ret'):
    '''Annualized Return over Max Drawdown
    INPUT 
        df: a dataframe containing a column of raw returns
    OUTPUT
        a float number of the Calmar ratio (a Calmar ratio of 0.50 is minimal. Anything over 1.0 is considered a pretty healthy risk adjusted return.)
    Reference: Terry W Young. Calmar ratio: A smoother tool. Futures, 20(1):40, 1991. '''

    calmar = annualized_return(df) / (max_drawdown(df) + 1e-5)
    return calmar

def sharpe_ratio(df: pd.DataFrame, ret_column = 'raw_ret', RF_column = 'RF'):
    '''Annualized Return - RF rate / Annualized Standand Deviation
    INPUT 
        df: a dataframe containing a column of raw returns and a column of risk-free rates
    OUTPUT
        a float number of the Sharpe ratio (a Sharpe ratio of 1.50 and higher is considered a good risk adjusted trading performance.)
    Reference: https://web.stanford.edu/~wfsharpe/ws/wi_perf.htm#:~:text=The%20annualized%20Sharpe%20Ratio%20is,the%20square%20root%20of%2012. '''

    # Calculate the annualized average daily excess return
    excess_returns = df[ret_column] - df[RF_column]
    ann_mean = excess_returns.mean() * 252  # there are about 252 business days per year

    # Calculate the annualized daily standard deviation
    ann_std_dev = excess_returns.std() * (252 ** 0.5) 

    return ann_mean / (ann_std_dev + 1e-4)

def sortino_ratio(df: pd.DataFrame, ret_column = 'raw_ret', RF_column = 'RF'):
    '''Annualized Return - RF rate / Annualized Downside Standard Deviation
    INPUT 
        df: a dataframe containing a column of raw returns and a column of risk-free rates
    OUTPUT
        a float number of the Sortino ratio
    Reference: https://www.investmacro.com/forex/2018/07/5-statistics-for-analyzing-your-trading-performance  '''

    # Calculate the annualized average daily excess return
    excess_returns = df[ret_column] - df[RF_column]
    ann_mean = excess_returns.mean() * 252  # there are about 252 business days per year

    # Calculate the annualized daily standard deviation of negative excess returns
    downside_excess_returns = excess_returns[excess_returns < 0]
    ann_downside_std_dev = downside_excess_returns.std() * (252 ** 0.5)
   
    return ann_mean / ann_downside_std_dev

def MRAR(df: pd.DataFrame, ret_column: str = 'raw_ret', RF_column: str = 'RF', gamma: float = 2.):
    ''' The Morningstar risk-adjusted rating
    INPUT
        df: a dataframe containing a column of raw returns and a column of risk-free rates
    OUTPUT
        a float number of the Morningstar performance measure
    Reference: Ingersoll J., Spiegel, M., Goetzmann W., and I. Welch (2007). Portfolio performance manipulation and manipulation-proof performance measures. Review of Financial Studies 20(5): 1503-1546
    '''
    # Calculate return ratios
    ratios = (1. + df[ret_column].values) / (1. + df[RF_column].values)

    return np.mean( ratios**(-gamma) ) ** (-252/gamma) - 1.


def cecpp(df1: pd.DataFrame, df2: pd.DataFrame, ret_column = 'raw_ret'):
    ''' Correlation between equity curve and perfect profit
    INPUT
        df1 and df2: two dataframes of raw returns from a trading strategy based on directional forecasts of price movements and the perfect trading strategy
    OUTPUT
        a float number of the CECPP
    '''
    return np.corrcoef(df1[ret_column].values, df2[ret_column].values)[0, 1]