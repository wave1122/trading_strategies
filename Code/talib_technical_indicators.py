# ======================================== Calculate All Technical Indicators and Trading Functions =========================================================== #
# ============================================= using Talib: https://github.com/mrjbq7/ta-lib =============================================================== #
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import re
from pathlib import Path
import glob2
from functools import reduce
import math
import datetime
from pandas.tseries.offsets import BDay, DateOffset
import time
import os
import shutil
import gc
import talib
from talib import MA_Type

# Calculate technical indicators
def get_tech_indicators(price_pd: pd.DataFrame, ticker: str, timeperiod: int, out_dir = 'e:/Copy/SCRIPTS/Forecast_Stocks/Data/'):
    """
    INPUT
        price_pd: a dataframe containing open, high, low, and close prices and volumes
        ticker: name of the stock ticker
    OUTPUT
        a dataframe containing the columns of the initial dataframe plus all technical indicators
    """
    df = price_pd.copy()

    # Get the dataframe columns
    open_price = price_pd.columns[1]
    high_price = price_pd.columns[2]
    low_price = price_pd.columns[3]
    close_price = price_pd.columns[4]
    volume = price_pd.columns[5]

    # Calculate Bollinger bands
    BBANDS_UP, BBANDS_MID, BBANDS_LOWER = talib.BBANDS(df[close_price].values, timeperiod = timeperiod)
    df[f'BBANDS_{timeperiod}'] = [1 if df[close_price].values[i] < BBANDS_LOWER[i] else -1 if df[close_price].values[i] > BBANDS_UP[i] else 0 for i in range(len(df[close_price]))]


    # Calculate the double EMAs
    df[f'DEMA_{timeperiod}'] = talib.DEMA(df[close_price].values, timeperiod = timeperiod)

    # Calculate EMAs
    df[f'EMA_{timeperiod}'] = talib.EMA(df[close_price].values, timeperiod = timeperiod)

    # Calculate Kaufman Adaptive MAs
    df[f'KAMA_{timeperiod}'] = talib.KAMA(df[close_price].values, timeperiod = timeperiod)

    # Calculate SMA
    df[f'SMA_{timeperiod}'] = talib.MA(df[close_price].values, timeperiod = timeperiod, matype = MA_Type.SMA)

    # Calculate the triple EMA
    df[f'TEMA_{timeperiod}'] = talib.TEMA(df[close_price].values, timeperiod = timeperiod)

    # Calculate Hilbert Transform - Instantaneous Trendline
    df[f'HT_TRENDLINE'] = talib.HT_TRENDLINE(df[close_price].values)

    # Calculate the Parabolic SAR indicator
    df[f'SAR'] = talib.SAR(df[high_price].values, df[low_price].values, acceleration=0, maximum=0)

    # Calculate the ADX
    df[f'ADX_{timeperiod}'] = talib.ADX(df[high_price].values, df[low_price].values, df[close_price].values, timeperiod = timeperiod)
    df[f'ADXR_{timeperiod}'] = talib.ADXR(df[high_price].values, df[low_price].values, df[close_price].values, timeperiod = timeperiod)

    # Calculate the  Absolute Price Oscillator
    df[f'APO_{timeperiod}'] = talib.APO(df[close_price].values, fastperiod = math.ceil(timeperiod/2), slowperiod = timeperiod, matype = MA_Type.SMA)

    # Calculate the AROON
    AROON_DOWN, AROON_UP = talib.AROON(df[high_price].values, df[low_price].values, timeperiod = timeperiod)
    df[f'AROON_{timeperiod}'] = [-1 if (AROON_DOWN[t] > AROON_UP[t]) and (AROON_DOWN[t-1] < AROON_UP[t-1]) else 1 \
                                                            if (AROON_DOWN[t] < AROON_UP[t]) and (AROON_DOWN[t-1] > AROON_UP[t-1]) else 0 for t in np.arange(len(df[close_price]))]


    df[f'AROONOSC_{timeperiod}'] = talib.AROONOSC(df[high_price].values, df[low_price].values, timeperiod = timeperiod)

    # Calculate the Balance of Power
    df[f'BOP_{timeperiod}'] = talib.BOP(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Commodity Channel Index
    df[f'CCI_{timeperiod}'] = talib.CCI(df[high_price].values, df[low_price].values, df[close_price].values, timeperiod = timeperiod)

    # Calculate the Chande Momentum Oscillator
    df[f'CMO_{timeperiod}'] = talib.CMO(df[close_price].values, timeperiod = timeperiod)

    # Calculate the Moving Average Convergence/Divergence
    MACD, MACDSIGNAL, MACDHIST = talib.MACD( df[close_price].values, fastperiod = math.ceil(timeperiod/2), \
                                                                                                                        slowperiod = timeperiod, signalperiod = math.ceil(timeperiod/3) )
    df[f'MACD_{timeperiod}'] = [1 if (MACD[t] > MACDSIGNAL[t]) and (MACD[t-1] < MACDSIGNAL[t-1]) else \
                                                    -1 if (MACD[t] < MACDSIGNAL[t]) and (MACD[t-1] > MACDSIGNAL[t-1]) else \
                                                    0 for t in np.arange(len(df[close_price]))]

    # Calculate the Money Flow Index
    df[f'MFI_{timeperiod}'] = talib.MFI(df[high_price].values, df[low_price].values, df[close_price].values, df[volume].values, timeperiod = timeperiod)

    # Calculate the Minus Directional Indicator and Movement
    df[f'MINUS_DI_{timeperiod}'] = talib.MINUS_DI(df[high_price].values, df[low_price].values, df[close_price].values, timeperiod = timeperiod)
    df[f'MINUS_DM_{timeperiod}'] = talib.MINUS_DM(df[high_price].values, df[low_price].values, timeperiod = timeperiod)

    # Calculate the momentum
    df[f'MOM_{timeperiod}'] = talib.MOM(df[close_price], timeperiod = timeperiod)

    # Calculate the Plus Directional Indicator and Movement
    df[f'PLUS_DI_{timeperiod}'] = talib.PLUS_DI(df[high_price].values, df[low_price].values, df[close_price].values, timeperiod = timeperiod)
    df[f'PLUS_DM_{timeperiod}'] = talib.PLUS_DM(df[high_price].values, df[low_price].values, timeperiod = timeperiod)

    # Calculate the Percentage Price Oscillator
    df[f'PPO_{timeperiod}'] = talib.PPO(df[close_price].values, fastperiod = math.ceil(timeperiod/2), slowperiod = timeperiod, matype = MA_Type.SMA)

    # Calculate the rate of change
    df[f'ROC_{timeperiod}'] = talib.ROC(df[close_price].values, timeperiod = timeperiod)

    # Calculate the Relative Strength Index
    df[f'RSI_{timeperiod}'] = talib.RSI(df[close_price].values, timeperiod = timeperiod)

    # Calculate the Stochastic Fast
    FASTK, FASTD = talib.STOCHF(df[high_price].values, df[low_price].values, df[close_price].values, fastk_period = timeperiod, fastd_period = math.ceil(timeperiod/2), \
                                                                                                                                                                                                                                 fastd_matype = MA_Type.SMA)

    # Calculate the Stochastic Slow
    SLOWK, SLOWD = talib.STOCH(df[high_price].values, df[low_price].values, df[close_price].values, fastk_period = timeperiod, slowk_period = math.ceil(timeperiod/2), \
                                                                                                        slowk_matype = MA_Type.SMA, slowd_period=math.ceil(timeperiod/2), slowd_matype = MA_Type.SMA)

    df[f'STOCH_{timeperiod}'] = [1 if (FASTD[t] > SLOWD[t]) and (FASTD[t-1] < SLOWD[t-1]) else -1 if (FASTD[t] < SLOWD[t]) and (FASTD[t-1] > SLOWD[t-1]) else 0 \
                                for t in np.arange(len(df[close_price]))]

    # Calculate the Stochastic Relative Strength Index
    df[f'STOCHRSI_{timeperiod}'], _ = talib.STOCHRSI(df[close_price].values, timeperiod = timeperiod, fastk_period = math.ceil(timeperiod/3), fastd_period = math.ceil(timeperiod/5), \
                                                                                                                                                                                                                                     fastd_matype = MA_Type.SMA)
    

    # Calculate the 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    df[f'TRIX_{timeperiod}'] = talib.TRIX(df[close_price].values, timeperiod = timeperiod)

    # Calculate the Ultimate Oscillator
    df[f'ULTOSC_{timeperiod}'] = talib.ULTOSC(df[high_price].values, df[low_price].values, df[close_price].values, timeperiod1 = math.ceil(timeperiod/3), \
                                                                                                                                                timeperiod2 = math.ceil(timeperiod/2), timeperiod3 = timeperiod)

    # Calculate Williams' %R
    df[f'WILLR_{timeperiod}'] = talib.WILLR(df[high_price].values, df[low_price].values, df[close_price].values, timeperiod = timeperiod)

    # Calculate the Chaikin A/D Line
    df[f'AD'] = talib.AD(df[high_price].values, df[low_price].values, df[close_price].values, df[volume].values)

    # Calculate the Chaikin A/D Oscillator
    df[f'ADOSC_{timeperiod}'] = talib.ADOSC(df[high_price].values, df[low_price].values, df[close_price].values, df[volume].values, fastperiod = math.ceil(timeperiod/2), slowperiod = timeperiod)

    # Calculate the On Balance Volume
    df[f'OBV'] = talib.OBV(df[close_price], df[volume])

    # Calculate the Average True Range
    df[f'ATR_{timeperiod}'] = talib.ATR(df[high_price].values, df[low_price].values, df[close_price].values, timeperiod = timeperiod)

    # Calculate the normalized Average True Range
    df[f'NATR_{timeperiod}'] = talib.NATR(df[high_price].values, df[low_price].values, df[close_price].values, timeperiod = timeperiod)

    # Calculate the True Range
    df[f'TRANGE'] = talib.TRANGE(df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Hilbert Transform - Dominant Cycle Period
    df[f'HT_DCPERIOD'] = talib.HT_DCPERIOD(df[close_price].values)

    # Calculate the Hilbert Transform - Dominant Cycle Phase
    df[f'HT_DCPHASE'] = talib.HT_DCPHASE(df[close_price].values)

    # # Calculate the Hilbert Transform - Phasor Components
    df[f'HT_PHASOR_inPHASE'], df[f'HT_PHASOR_QUADR']  = talib.HT_PHASOR(df[close_price].values)

    # Calculate the Hilbert Transform - SineWave
    df[f'HT_SINE_LOWER'],  df[f'HT_SINE_UPPER']= talib.HT_SINE(df[close_price].values)

    # Calculate the Hilbert Transform - Trend vs Cycle Mode
    df[f'HT_TRENDMODE'] = talib.HT_TRENDMODE(df[close_price].values)

    # Calculate the rolling standard deviations
    df[f'PR_STDDEV_{timeperiod}'] = talib.STDDEV(df[close_price].values, timeperiod = timeperiod)

    df.to_csv(os.path.join(out_dir, '%s_tech_ids.csv' % ticker), index = False, header = True)
    return df

# Calculate pattern recognition functions
def get_pattern_recognitions(price_pd: pd.DataFrame, ticker: str, out_dir = 'e:/Copy/SCRIPTS/Forecast_Stocks/Data/'):
    """
    INPUT
        price_pd: a dataframe containing open, high, low, and close prices and volumes
        ticker: name of the stock ticker
    OUTPUT
        a dataframe containing the columns of the initial dataframe plus all pattern recognition functions
    """
    df = price_pd.copy()

    # Get the dataframe columns
    open_price = price_pd.columns[1]
    high_price = price_pd.columns[2]
    low_price = price_pd.columns[3]
    close_price = price_pd.columns[4]
    volume = price_pd.columns[5]

    # Calculate the Two Crows pattern
    df['CDL2CROWS'] = talib.CDL2CROWS(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Three Black Crows pattern
    df['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Three Inside Up/Down pattern
    df['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Three Stars In The South pattern
    df['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Three Advancing White Soldiers pattern
    df['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Abandoned Baby pattern
    df['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Advance Block pattern
    df['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Belt-hold pattern
    df['CDLBELTHOLD'] = talib.CDLBELTHOLD(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Breakaway pattern
    df['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Closing Marubozu pattern
    df['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Concealing Baby Swallow pattern
    df['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Counterattack pattern
    df['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Dark Cloud Cover pattern
    df['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Doji pattern
    df['CDLDOJI'] = talib.CDLDOJI(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Doji Star pattern
    df['CDLDOJISTAR'] = talib.CDLDOJISTAR(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Dragonfly Doji pattern
    df['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Engulfing pattern
    df['CDLENGULFING'] = talib.CDLENGULFING(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Evening Doji Star pattern
    df['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Evening Star pattern
    df['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Up/Down-gap side-by-side white lines pattern
    df['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Gravestone Doji pattern
    df['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Hammer pattern
    df['CDLHAMMER'] = talib.CDLHAMMER(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Hanging Man pattern
    df['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Harami pattern
    df['CDLHARAMI'] = talib.CDLHARAMI(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate Harami Cross pattern
    df['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the High-Wave Candle pattern
    df['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Hikkake pattern
    df['CDLHIKKAKE'] = talib.CDLHIKKAKE(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the modified Hikkake pattern
    df['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Homing Pigeon pattern
    df['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Identical Three Crows pattern
    df['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the In-Neck pattern
    df['CDLINNECK'] = talib.CDLINNECK(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Inverted Hammer pattern
    df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Kicking pattern
    df['CDLKICKING'] = talib.CDLKICKING(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Kicking pattern - bull/bear determined by the longer marubozu
    df['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Ladder Bottom pattern
    df['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Long Legged Doji pattern
    df['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Long Line Candle pattern
    df['CDLLONGLINE'] = talib.CDLLONGLINE(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Marubozu pattern
    df['CDLMARUBOZU'] = talib.CDLMARUBOZU(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Matching Low pattern
    df['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Mat Hold pattern
    df['CDLMATHOLD'] = talib.CDLMATHOLD(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Morning Doji Star pattern
    df['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Morning Star pattern
    df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the On-Neck pattern
    df['CDLONNECK'] = talib.CDLONNECK(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Piercing pattern
    df['CDLPIERCING'] = talib.CDLPIERCING(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Rickshaw Man pattern
    df['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Rising/Falling Three Methods pattern
    df['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Separating Lines pattern
    df['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Shooting Star pattern
    df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Short Line Candle pattern
    # df['CDLSHORTLINE'] = talib.CDLSHORTLINE(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Spinning Top pattern
    df['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Stalled pattern
    df['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Stick Sandwich pattern
    df['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Takuri pattern (Dragonfly Doji with very long lower shadow)
    df['CDLTAKURI'] = talib.CDLTAKURI(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Tasuki Gap pattern
    df['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Thrusting pattern
    df['CDLTHRUSTING'] = talib.CDLTHRUSTING(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Tristar pattern
    df['CDLTRISTAR'] = talib.CDLTRISTAR(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Unique 3 River pattern
    df['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Upside Gap Two Crows pattern
    df['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)

    # Calculate the Upside/Downside Gap Three Methods pattern
    df['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(df[open_price].values, df[high_price].values, df[low_price].values, df[close_price].values)
    
    df.to_csv(os.path.join(out_dir, '%s_patterns.csv' % ticker), index = False, header = True)
    return df