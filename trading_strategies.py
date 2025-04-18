# ================================================== Implement Trading Stragegies based on Directional Forecasts ================================================ #
import os

# Use CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['NUMEXPR_MAX_THREADS'] = '30'

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels import robust
from statistics import median
import sys
import gc
import math
from varname import nameof

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import datetime
import time
import psutil
import multiprocessing as multi
from itertools import product, repeat
from functools import partial

# import a tupling module
from typing import Tuple


# ##### Set the current working directory
# path="e:\\Copy\\SCRIPTS\\Forecast_Stocks\\Code\\"
# os.chdir(path)

def trade_fixed_trans_cost( df: pd.DataFrame, 
                                            W0: float, # initial wealth
                                            trans_cost: float, # a fixed transaction cost
                                            trans_date_column = 'trans_date',
                                            price_column = 'price',
                                            proba_forecast_column = 'proba_forecast',
                                            RF_column = 'RF',
                                            ) -> Tuple[np.float32, np.float32, pd.DataFrame]:
    ''' Calculate the equity curve of a trading strategy [with fixed transaction cost] which buys or sells a risky asset based on the prediction of the directional movement of its future price.
     INPUT
        df: a dataframe containing a column of closing prices, a column of risk-free rates, and a column of probability forecasts of the directional movement of future asset prices
        W0: the initial wealth
        trans_cost: a fixed amount of transaction cost
    OUTPUT 
        the wealth of the buy-and-hold strategy, its total return, and the equity curve
    '''
    df = df.copy() # create a deep copy of the original dataframe

    T = df.shape[0] # the number of trading days
    weights = np.zeros(shape = T-1)
    wealths = np.zeros(shape = T-1)

    trans_dates, stock_prices, rf_rates = [], [], []
    trans_dates.append(df.iloc[0][trans_date_column])
    stock_prices.append(df.iloc[0][price_column])
    rf_rates.append(df.iloc[0][RF_column])

    # Initialize the strategy by buying stock
    weights[0] = (W0 - trans_cost) / df.iloc[0][price_column]
    wealths[0] = W0 # this is the wealth prior to the buy transaction

    for t in np.arange(1, T-1):
        trans_dates.append(df.iloc[t][trans_date_column])
        stock_prices.append(df.iloc[t][price_column])
        rf_rates.append(df.iloc[t][RF_column])

        if df.iloc[t+1][proba_forecast_column] >= 0.5: # if the price is going up
            if weights[t-1] > 0.: # the last position was in stock
                weights[t] = weights[t-1] # continue to hold the stock
                wealths[t] = weights[t] * df.iloc[t][price_column]  - trans_cost # this is the wealth assuming that the stock position is liquidated at the point in time
            else:
                wealths[t] = wealths[t-1] * (1. + df.iloc[t-1][RF_column]) # wealth accrued by holding cash till this point in time
                weights[t] = (wealths[t] - trans_cost) / df.iloc[t][price_column] # buy the stock
        else: # if the price is going down
            if weights[t-1] > 0.: # the last position was in stock
                weights[t] = 0. # sell the stock
                wealths[t] = weights[t-1] * df.iloc[t][price_column] - trans_cost
            else:
                weights[t] = 0. # continue to hold cash
                wealths[t] = wealths[t-1] * (1. + df.iloc[t-1][RF_column]) # wealth accrued by holding cash till this point in time

    # Liquidate the stock position in the end
    weights[T-2] = 0.

    perform_df = pd.DataFrame({'trans_date': trans_dates, 'price': stock_prices, 'RF': rf_rates, 'weights': weights, 'total_wealth': wealths})
    perform_df['raw_ret'] = perform_df['total_wealth'].pct_change()

    # Compute the arithmetic return of the buy-and-hold strategy
    wealth_bh = weights[0] * df.iloc[T-2][price_column] - trans_cost
    ret_bh = (wealth_bh - W0) / W0
    
    return wealth_bh, ret_bh, perform_df

def trade_variable_trans_cost(   df: pd.DataFrame, 
                                                    W0: float, # initial wealth
                                                    trans_cost: float, # a variable transaction cost in between [0., 1.]
                                                    trans_date_column = 'trans_date',
                                                    price_column = 'price',
                                                    proba_forecast_column = 'proba_forecast',
                                                    RF_column = 'RF',
                                                ) -> Tuple[np.float32, np.float32, pd.DataFrame]:
    ''' Calculate the equity curve of a trading strategy [with variable transaction cost] which buys or sells a risky asset based on the prediction of the directional movement of its future price.
     INPUT
        df: a dataframe containing a column of closing prices, a column of risk-free rates, and a column of probability forecasts of the directional movement of future asset prices
        W0: the initial wealth
        trans_cost: a variable transaction cost in between [0., 1.]
    OUTPUT 
        the wealth of the buy-and-hold strategy, its total return, and the equity curve
    '''
    assert(trans_cost >= 0 and trans_cost <= 1.), 'the transaction cost must be greater than zero and less than one!'

    df = df.copy() # create a deep copy of the original dataframe

    T = df.shape[0] # the number of trading days
    weights = np.zeros(shape = T-1)
    wealths = np.zeros(shape = T-1)

    trans_dates, stock_prices, rf_rates = [], [], []
    trans_dates.append(df.iloc[0][trans_date_column])
    stock_prices.append(df.iloc[0][price_column])
    rf_rates.append(df.iloc[0][RF_column])

    # start the strategy by buying the stock
    weights[0] = W0 / ( df.iloc[0][price_column] * (1. + trans_cost) )
    wealths[0] = W0

    for t in np.arange(1, T-1):
        trans_dates.append(df.iloc[t][trans_date_column])
        stock_prices.append(df.iloc[t][price_column])
        rf_rates.append(df.iloc[t][RF_column])

        if df.iloc[t+1][proba_forecast_column] >= 0.5: # if price is going up
            if weights[t-1] > 0.: # if the last position was in stock
                weights[t] = weights[t-1] # keep on holding the stock
                wealths[t] = weights[t-1] * df.iloc[t][price_column] *(1. - trans_cost)  # this is the wealth assuming that the stock position is liquidated at the point in time
            else: # if the last position was in cash
                wealths[t] = wealths[t-1] * (1. + df.iloc[t-1][RF_column])  # wealth accrued by holding cash
                weights[t] = wealths[t] / ( df.iloc[t][price_column] * (1. + trans_cost) ) # buy stock
        else: # if price is going down
            if weights[t-1] > 0.: # if the last position was in stock
                weights[t] = 0. # sell the stock
                wealths[t] = weights[t-1] * df.iloc[t][price_column] * (1. - trans_cost)
            else: # if the last position was in cash
                weights[t] = 0. # continue to hold cash
                wealths[t] = wealths[t-1] * (1. + df.iloc[t-1][RF_column])  # wealth accrued by holding cash

    # Liquidate the stock position in the end
    weights[T-2] = 0.

    perform_df = pd.DataFrame({'trans_date': trans_dates, 'price': stock_prices, 'RF': rf_rates, 'weights': weights, 'total_wealth': wealths})
    perform_df['raw_ret'] = perform_df['total_wealth'].pct_change()

    # Compute the arithmetic return of the buy-and-hold strategy
    wealth_bh = weights[0] * df.iloc[T-2][price_column] * (1. - trans_cost)
    ret_bh = (wealth_bh - W0) / W0
    
    return wealth_bh, ret_bh, perform_df


def perfect_profit_fixed_trans_cost(  df: pd.DataFrame, 
                                                            W0: float, # initial wealth
                                                            trans_cost: float, # a fixed transaction cost
                                                            trans_date_column = 'trans_date',
                                                            price_column = 'price',
                                                            RF_column = 'RF',
                                                        ) -> Tuple[np.float32, np.float32, pd.DataFrame]:
    ''' Calculate the perfect equity curve (i.e., an equity curve assuming that the signs of one-period ahead returns are known)
        with a fixed transaction cost
    INPUT
        df: a dataframe containing a column of closing prices and a column of risk-free rates
        W0: the initial wealth
        trans_cost: a fixed amount of transaction cost
    OUTPUT 
        the wealth of the buy-and-hold strategy, its total return, and the perfect equity curve
    '''
    df = df.copy() # create a deep copy of the original dataframe

    T = df.shape[0]
    weights = np.zeros(shape = T-1)
    wealths = np.zeros(shape = T-1)

    raw_returns = df[price_column].pct_change()

    trans_dates, stock_prices, rf_rates = [], [], []
    trans_dates.append(df.iloc[0][trans_date_column])
    stock_prices.append(df.iloc[0][price_column])
    rf_rates.append(df.iloc[0][RF_column])

    # Initialize a position in stock if the price is going up
    wealths[0] = W0
    if raw_returns.iloc[1] >= 0:
        weights[0] = (W0 - trans_cost) / df.iloc[0][price_column]
    else:
        weights[0] = 0.
    
    # Calculate the weight of the buy-and-hold strategy
    weight_bh = (wealths[0] - trans_cost) / df.iloc[0][price_column]

    for t in np.arange(1, T-1):
        trans_dates.append(df.iloc[t][trans_date_column])
        stock_prices.append(df.iloc[t][price_column])
        rf_rates.append(df.iloc[t][RF_column])

        if raw_returns.iloc[t+1] >= 0: # price is going up
            if weights[t-1] > 0.: # the last position was in stock
                weights[t] = weights[t-1] # continue to hold equity
                wealths[t] = weights[t] * df.iloc[t][price_column]  - trans_cost # this is the wealth assuming that the stock position is liquidated at the point in time
            else: # the last position was in cash
                wealths[t] = wealths[t-1] * (1. + df.iloc[t-1][RF_column]) # the wealth accrued till this point in time
                weights[t] = (wealths[t] - trans_cost) / df.iloc[t][price_column] # buy stock
        else: # price is going down
            if weights[t-1] > 0.: # the last position is in stock
                weights[t] = 0. # liquidate the stock position
                wealths[t] = weights[t-1] * df.iloc[t][price_column]  - trans_cost 
            else: # the last position was in cash
                weights[t] = 0. # continue to hold cash
                wealths[t] = wealths[t-1] * (1. + df.iloc[t-1][RF_column]) # the wealth accured till this point in time

    # liquidate the stock position in the end
    weights[T-2] = 0.

    perform_df = pd.DataFrame({'trans_date': trans_dates, 'price': stock_prices, 'RF': rf_rates, 'weights': weights, 'total_wealth': wealths})
    perform_df['raw_ret'] = perform_df['total_wealth'].pct_change()

    # Compute the arithmetic return of the buy-and-hold strategy
    wealth_bh = weight_bh * df.iloc[T-2][price_column] - trans_cost
    ret_bh = (wealth_bh - W0) / W0

    return wealth_bh, ret_bh, perform_df


def perfect_profit_variable_trans_cost( df: pd.DataFrame, 
                                                                W0: float, # initial wealth
                                                                trans_cost: float, # a variable transaction cost in between [0., 1.]
                                                                trans_date_column = 'trans_date',
                                                                price_column = 'price',
                                                                RF_column = 'RF',
                                                            ) -> Tuple[np.float32, np.float32, pd.DataFrame]:
    ''' Calculate the perfect equity curve (i.e., an equity curve assuming that the signs of one-period ahead returns are known)
        with a variable transaction cost
    INPUT
        df: a dataframe containing a column of closing prices and a column of risk-free rates
        W0: the initial wealth
        trans_cost: a variable transaction cost in between [0., 1.]
    OUTPUT 
        the wealth of the buy-and-hold strategy, its total return, and the perfect equity curve
    '''
    assert(trans_cost >= 0 and trans_cost <= 1.), 'the transaction cost must be greater than zero and less than one!'

    df = df.copy() # make a deep copy of the original dataframe

    T = df.shape[0]
    weights = np.zeros(shape = T-1)
    wealths = np.zeros(shape = T-1)

    raw_returns = df[price_column].pct_change()

    trans_dates, stock_prices, rf_rates = [], [], []
    trans_dates.append(df.iloc[0][trans_date_column])
    stock_prices.append(df.iloc[0][price_column])
    rf_rates.append(df.iloc[0][RF_column])

    # Initialize a position in stock if the price is going up
    wealths[0] = W0
    if raw_returns.iloc[1] >= 0:
        weights[0] = W0 / ( df.iloc[0][price_column]*(1. + trans_cost) )
    else:
        weights[0] = 0.
    
    # Calculate the weight of the buy-and-hold strategy
    weight_bh = wealths[0] / ( df.iloc[0][price_column]*(1. + trans_cost) )

    for t in np.arange(1, T-1):
        trans_dates.append(df.iloc[t][trans_date_column])
        stock_prices.append(df.iloc[t][price_column])
        rf_rates.append(df.iloc[t][RF_column])

        if raw_returns.iloc[t+1] >= 0: # price is going up
            if weights[t-1] > 0.: # the last position was in stock
                weights[t] = weights[t-1] # continue to hold equity
                wealths[t] = weights[t] * df.iloc[t][price_column] * (1. - trans_cost) # this is the wealth assuming that the stock position is liquidated at the point in time
            else: # the last position was in cash
                wealths[t] = wealths[t-1] * (1. + df.iloc[t-1][RF_column]) # the wealth accrued till this point in time
                weights[t] = wealths[t]  / ( df.iloc[t][price_column]*(1. + trans_cost) ) # buy stock
        else: # price is going down
            if weights[t-1] > 0.: # the last position is in stock
                weights[t] = 0. # liquidate the stock position
                wealths[t] = weights[t-1] * df.iloc[t][price_column]  * (1. - trans_cost)
            else: # the last position was in cash
                weights[t] = 0. # continue to hold cash
                wealths[t] = wealths[t-1] * (1. + df.iloc[t-1][RF_column]) # the wealth accured till this point in time

    # liquidate the stock position in the end
    weights[T-2] = 0.

    perform_df = pd.DataFrame({'trans_date': trans_dates, 'price': stock_prices, 'RF': rf_rates, 'weights': weights, 'total_wealth': wealths})
    perform_df['raw_ret'] = perform_df['total_wealth'].pct_change()

    # Compute the arithmetic return of the buy-and-hold strategy
    wealth_bh = weight_bh * df.iloc[T-2][price_column] * (1. - trans_cost)
    ret_bh = (wealth_bh - W0) / W0

    return wealth_bh, ret_bh, perform_df
