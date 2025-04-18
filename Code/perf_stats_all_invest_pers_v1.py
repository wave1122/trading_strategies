# ==================================== Calculate the medians, IQRs, means, standard deviations, t-statistics of performance metrics ============================================= #
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, List
import time
import os

import statsmodels.api as sm
from scipy import stats

# Define a function to round a number to a given number of decimal places
def round_number(x, decimals = 4):
    ''' Round a number to a given number of decimal places
    '''
    try:
        return round(x, decimals)
    except:
        return x

# Define function to calculate the interquartile range
def iqr(x: np.ndarray):
    ''' Calculate the IQR of a high-dimensional array
    '''
    q75 = np.nanpercentile(x, 75, axis = 0) # compute the 75% quantile
    q25 = np.nanpercentile(x, 25, axis = 0)  # compute the 25% quantile
    return q75 - q25

# Define function to calculate the median
def med(x: np.ndarray):
    ''' Calculate the median of a high-dimensional array
    '''
    return np.nanpercentile(x, 50, axis = 0)

# Define function to calculate the mean
def mean(x: np.ndarray):
    ''' Calculate the mean of a high-dimensional array
    '''
    return np.nanmean(x, axis = 0)

# Define function to calculate the standard deviation
def std(x: np.ndarray):
    ''' Calculate the standard deviation of a high-dimensional array
    '''
    return np.nanstd(x, axis = 0)

# Calculate the Newey and West's (1987) standard error
def NWstd(x: np.ndarray):
    """  Calculate the Newey and West's (1987) standard error for each column of an array.
    INPUT
        x: a 2-dimensional numpy array
    """    
    N = x.shape[1]
    stds = []
    for i in range(N):
        xi = x[:, i]
        xi = xi[~np.isnan(xi)]
        # print('x = %s' % x)
        
        T = len(xi)
        ones = np.array([1. for _ in range(T)])

        reg_model = sm.OLS(endog = xi, exog = ones) # an intercept is not included by default
        reg = reg_model.fit(cov_type='HAC', cov_kwds={'maxlags':6})
        # print(reg.summary() )
        stds.append(reg.bse.squeeze())

    return np.array(stds)

def summary_stats (use_model: str = 'LGBM_SPY_all_vars', 
					tau: int = 1,  # a forecast horizon
					loss_fn: str = 'Brier', 
					score_fn: str = 'CECPP_fixed_trans_cost'):
    ''' Calculate the means, standard deviations, Newey and West's (1987) standard errors, medians, and IQRs of the performance metrics over multiple investment periods
    '''
    
    # List all filepaths in a folder
    folder_path = Path(f'../Results/{use_model}/loss_fn={loss_fn}/score_fn={score_fn}/tau={tau}/performance/')
    file_paths = sorted([str(path) for path in folder_path.glob('*.csv')])
    print(file_paths)

    # Create a folder for output data
    out_dir = folder_path.joinpath('descriptive_stats')

    if not os.path.exists( out_dir ):
        # Create the directory if it does not exist .
        os.makedirs( out_dir )

    perf_df = pd.read_csv(file_paths[0], encoding='utf-8', sep = ',', low_memory=False, header = 0, skiprows = 0, skipinitialspace=True, parse_dates = ['start_date', 'end_date'])
    perf_df.drop(columns = ['start_date', 'end_date', 'ratio_profit_over_total_loss', 'annualized_return',	'annualized_return_bh'], inplace = True)

    get_hper = re.search(r'(?<=hper_)\d+(?=_)', file_paths[0], flags=re.IGNORECASE | re.VERBOSE).group()
    get_strategy = re.search(r'(?<=1000_)\w+(?=_\d+)', file_paths[0], flags=re.IGNORECASE | re.VERBOSE).group()
    get_trans_cost = re.search(r'(?<=\_)\d+\.?\d*(?=\.csv)', file_paths[0], flags=re.IGNORECASE | re.VERBOSE).group()

    median_df = pd.DataFrame(data = med(perf_df.values).reshape(-1, 16), columns = perf_df.columns)
    median_df.insert(loc = 0, column = 'strategy', value = get_strategy)
    median_df.insert( loc = 1, column = 'holding_per', value = int(get_hper) )
    median_df.insert( loc = 2, column = 'trans_cost', value = float(get_trans_cost) )

    iqr_df = pd.DataFrame(data = iqr(perf_df.values).reshape(-1, 16), columns = perf_df.columns)
    iqr_df.insert(loc = 0, column = 'strategy', value = get_strategy)
    iqr_df.insert( loc = 1, column = 'holding_per', value = int(get_hper) )
    iqr_df.insert( loc = 2, column = 'trans_cost', value = float(get_trans_cost) )

    mean_df = pd.DataFrame(data = mean(perf_df.values).reshape(-1, 16), columns = perf_df.columns)
    mean_df.insert(loc = 0, column = 'strategy', value = get_strategy)
    mean_df.insert( loc = 1, column = 'holding_per', value = int(get_hper) )
    mean_df.insert( loc = 2, column = 'trans_cost', value = float(get_trans_cost) )

    std_df = pd.DataFrame(data = std(perf_df.values).reshape(-1, 16), columns = perf_df.columns)
    std_df.insert(loc = 0, column = 'strategy', value = get_strategy)
    std_df.insert( loc = 1, column = 'holding_per', value = int(get_hper) )
    std_df.insert( loc = 2, column = 'trans_cost', value = float(get_trans_cost) )
    
    NWstd_df = pd.DataFrame(data = NWstd(perf_df.values).reshape(-1, 16), columns = perf_df.columns)
    NWstd_df.insert(loc = 0, column = 'strategy', value = get_strategy)
    NWstd_df.insert(loc = 1, column = 'holding_per', value = int(get_hper) )
    NWstd_df.insert(loc = 2, column = 'trans_cost', value = float(get_trans_cost) )

    for i, file_path in enumerate(file_paths[1:], start =1):
        get_hper = re.search(r'(?<=hper_)\d+(?=_)', file_path, flags=re.IGNORECASE | re.VERBOSE).group()
        get_strategy = re.search(r'(?<=1000_)\w+(?=_\d+)', file_path, flags=re.IGNORECASE | re.VERBOSE).group()
        get_trans_cost = re.search(r'(?<=\_)\d+\.?\d*(?=\.csv)', file_path, flags=re.IGNORECASE | re.VERBOSE).group()

        perf_df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory=False, header = 0, skiprows = 0, skipinitialspace=True, parse_dates = ['start_date', 'end_date'])
        # display(perf_df.head() )

        perf_df.drop(columns = ['start_date', 'end_date', 'ratio_profit_over_total_loss', 'annualized_return',	'annualized_return_bh'], inplace = True)
        median_df.loc[i] = [get_strategy, int(get_hper), float(get_trans_cost)] + med(perf_df.values).tolist()
        iqr_df.loc[i] = [get_strategy, int(get_hper), float(get_trans_cost)] + iqr(perf_df.values).tolist()
        mean_df.loc[i] = [get_strategy, int(get_hper), float(get_trans_cost)] + mean(perf_df.values).tolist()
        std_df.loc[i] = [get_strategy, int(get_hper), float(get_trans_cost)] + std(perf_df.values).tolist()
        NWstd_df.loc[i] = [get_strategy, int(get_hper), float(get_trans_cost)] + NWstd(perf_df.values).tolist()

    median_df = median_df.apply(round_number).sort_values(by =['strategy', 'holding_per', 'trans_cost'])
    median_df.to_csv(os.path.join(out_dir, f'median.csv'), index = False, header = True)

    iqr_df = iqr_df.apply(round_number).sort_values(by =['strategy', 'holding_per', 'trans_cost'])
    iqr_df.to_csv(os.path.join(out_dir, f'iqr.csv'), index = False, header = True)

    mean_df = mean_df.apply(round_number).sort_values(by =['strategy', 'holding_per', 'trans_cost'])
    mean_df.to_csv(os.path.join(out_dir, f'mean.csv'), index = False, header = True)

    std_df = std_df.apply(round_number).sort_values(by =['strategy', 'holding_per', 'trans_cost'])
    std_df.to_csv(os.path.join(out_dir, f'std.csv'), index = False, header = True)
    
    NWstd_df = NWstd_df.apply(round_number).sort_values(by =['strategy', 'holding_per', 'trans_cost'])
    NWstd_df.to_csv(os.path.join(out_dir, f'NWstd.csv'), index = False, header = True)

    # Report the standard deviations next to the means
    n_rows, n_cols = mean_df.shape
    mean_std_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(n_cols):
            if j < 3:
                mean_std_list[i].append( f'{mean_df.values[i, j]}' )
            else:
                mean_std_list[i].append( f'{mean_df.values[i, j]} | ({std_df.values[i, j]})' )
    
    mean_std_df = pd.DataFrame(mean_std_list, columns = mean_df.columns).sort_values(by =['strategy', 'holding_per', 'trans_cost'])
    mean_std_df.to_csv(os.path.join(out_dir, f'mean_std.csv'), index = False, header = True)
    
    # Report the t-statistics next to the means
    mean_tstat_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(n_cols):
            if j < 3:
                mean_tstat_list[i].append( f'{mean_df.values[i, j]}' )
            else:
                mean_tstat_list[i].append( f'{mean_df.values[i, j]} | ({ round_number(mean_df.values[i, j] / (1E-4 + NWstd_df.values[i, j])) })' )
    
    mean_tstat_df = pd.DataFrame(mean_tstat_list, columns = mean_df.columns).sort_values(by =['strategy', 'holding_per', 'trans_cost'])
    mean_tstat_df.to_csv(os.path.join(out_dir, f'mean_tstat.csv'), index = False, header = True)

    # Report the interquartile ranges next to the medians
    n_rows, n_cols = median_df.shape
    median_iqr_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(n_cols):
            if j < 3:
                median_iqr_list[i].append( f'{median_df.values[i, j]}' )
            else:
                median_iqr_list[i].append( f'{median_df.values[i, j]} | ({iqr_df.values[i, j]})' )
    
    median_iqr_df = pd.DataFrame(median_iqr_list, columns = median_df.columns).sort_values(by =['strategy', 'holding_per', 'trans_cost'])
    median_iqr_df.to_csv(os.path.join(out_dir, f'median_iqr.csv'), index = False, header = True)

    return True

def compare_score_fns ( use_model: str = 'LGBM_SPY_all_vars', 
                                        tau: int = 1,  # a forecast horizon
                                        loss_fn: str = 'Brier', 
                                        score_fns: List[str] =  ['Accuracy', 'AUC',  'Gain_to_pain_ratio_fixed_trans_cost', 'Calmar_ratio_fixed_trans_cost', \
                                                                                'Sharpe_ratio_fixed_trans_cost',  'Sortino_ratio_fixed_trans_cost',  'CECPP_fixed_trans_cost'], 
                                        perf_metric: str = 'cecpp'):
    ''' Compare the means (standard deviations), means (t-statistics),  and medians (IQRs) of a performance metric over multiple investment periods 
        across various score functions for a given loss function
    INPUT
        use_model: the output from a ML model
        loss_fn: a loss function used for training
        score_fns: a list of scoring functions used for cross validation
        perf_metric: a metric used to evaluate the performance of a trading strategy (e.g., average_number_of_trades, percentage_of_winning_trades,	largest_raw_return,	
            smallest_raw_return,	ratio_win_loss, max_number_of_consecutive_winners, max_number_of_consecutive_losers, annualized_excess_return, annualized_standard_deviation,
            max_drawdown,	Schwager_gain-to-pain_ratio,	Calmar_ratio,	Sharpe_ratio,	Sortino_ratio,	cecpp,	mrar)
    OUTPUT
        dataframes of means (standard deviations), means (t-statistics), and medians (IQRs)
    '''
    # Read data as a list of dataframes
    folder_path = Path(f'../Results/{use_model}/loss_fn={loss_fn}/')
    try:
        use_strategy = re.search(r'(?<=ratio\_)\w+', score_fns[2], flags=re.IGNORECASE | re.VERBOSE).group()
    except:
        try:
            use_strategy = re.search(r'(?<=ratio\_)\w+', score_fns[0], flags=re.IGNORECASE | re.VERBOSE).group()
        except:
            pass
        pass
    

    list_mean_std_df, list_mean_tstat_df, list_median_iqr_df = [], [], []
    for i in np.arange(len(score_fns)):
        file_path = folder_path.joinpath(f'score_fn={score_fns[i]}/tau={tau}/performance/descriptive_stats/mean_std.csv')
        df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True)
        if i < 2:
            df = df[df['strategy'] == use_strategy]
        list_mean_std_df.append(df)
        
        file_path = folder_path.joinpath(f'score_fn={score_fns[i]}/tau={tau}/performance/descriptive_stats/mean_tstat.csv')
        df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True)
        if i < 2:
            df = df[df['strategy'] == use_strategy]
        list_mean_tstat_df.append(df)

        file_path = folder_path.joinpath(f'score_fn={score_fns[i]}/tau={tau}/performance/descriptive_stats/median_iqr.csv')
        df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True)
        if i < 2:
            df = df[df['strategy'] == use_strategy]
        list_median_iqr_df.append(df)

    # display(list_df[0].head() )
    # display(list_df[1].head() )

    # Create a folder for output data
    out_dir = folder_path.joinpath('descriptive_stats')
    if not os.path.exists( out_dir ):
        # Create the directory if it does not exist .
        os.makedirs( out_dir )

    # Report the means (standard deviations) of a performance metric across score functions
    n_rows = list_mean_std_df[0].shape[0]
    mean_std_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(len(score_fns) + 3):
            if j < 3:
                mean_std_list[i].append(list_mean_std_df[0].values[i, j])
            else:
                mean_std_list[i].append(list_mean_std_df[j-3][perf_metric].values[i])

    mean_std_df = pd.DataFrame(mean_std_list, columns = list_mean_std_df[0].columns.tolist()[0:3] +  score_fns)
    mean_std_df.to_csv(os.path.join(out_dir, f'mean_std_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv'), index = False, header = True)
    
    # Report the means (t-statistics) of a performance metric across score functions
    n_rows = list_mean_tstat_df[0].shape[0]
    mean_tstat_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(len(score_fns) + 3):
            if j < 3:
                mean_tstat_list[i].append(list_mean_tstat_df[0].values[i, j])
            else:
                mean_tstat_list[i].append(list_mean_tstat_df[j-3][perf_metric].values[i])

    mean_tstat_df = pd.DataFrame(mean_tstat_list, columns = list_mean_tstat_df[0].columns.tolist()[0:3] +  score_fns)
    mean_tstat_df.to_csv(os.path.join(out_dir, f'mean_tstat_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv'), index = False, header = True)

     # Report the medians (IQRs) of a performance metric across score functions
    n_rows = list_median_iqr_df[0].shape[0]
    median_iqr_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(len(score_fns) + 3):
            if j < 3:
                median_iqr_list[i].append(list_median_iqr_df[0].values[i, j])
            else:
                median_iqr_list[i].append(list_median_iqr_df[j-3][perf_metric].values[i])

    median_iqr_df = pd.DataFrame(median_iqr_list, columns = list_median_iqr_df[0].columns.tolist()[0:3] +  score_fns)
    median_iqr_df.to_csv(os.path.join(out_dir, f'median_iqr_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv'), index = False, header = True)
    
    return True

def compare_loss_fns (use_model: str = 'LGBM_SPY_all_vars', 
                                    tau: int = 1, # a forecast horizon
                                    loss_fns: List[str] = ['As1', 'As2'], 
                                    score_fn: str = 'AUC', 
                                    perf_metric: str = 'cecpp'):
    ''' Compare the means (standard deviations), means (t-statistics), and medians (IQRs) of a performance metric over multiple investment periods 
        and loss functions for a given scoring function
    INPUT
        use_model: the output from a ML model
        loss_fns: a list of loss functions used for training (i.e., 'CE', 'Brier', 'Boost', 'As1', 'As2')
        score_fn: a scoring function used for cross validation (i.e.,  'Accuracy', 'AUC', 'Gain_to_pain_ratio_fixed_trans_cost', 'Gain_to_pain_ratio_variable_trans_cost', 
            'Calmar_ratio_fixed_trans_cost', 'Calmar_ratio_variable_trans_cost', 'Sharpe_ratio_fixed_trans_cost', 'Sharpe_ratio_variable_trans_cost', \
            'Sortino_ratio_fixed_trans_cost', 'Sortino_ratio_variable_trans_cost', 'CECPP_fixed_trans_cost', 'CECPP_variable_trans_cost')
        perf_metric: a metric used to evaluate the performance of a trading strategy (e.g., average_number_of_trades, percentage_of_winning_trades,	largest_raw_return,	
            smallest_raw_return,	ratio_win_loss, max_number_of_consecutive_winners, max_number_of_consecutive_losers, annualized_excess_return, annualized_standard_deviation,
            max_drawdown,	Schwager_gain-to-pain_ratio,	Calmar_ratio,	Sharpe_ratio,	Sortino_ratio,	cecpp,	mrar)
    OUTPUT
        dataframes of means (standard deviations), means (t-statistics), and medians (IQRs)
    '''
    # Read data as a list of dataframes
    folder_path = Path(f'../Results/{use_model}/')
    try:
        use_strategy = re.search(r'(?<=ratio\_|CECPP\_)\w+', score_fn, flags=re.IGNORECASE | re.VERBOSE).group()
    except:
        pass

    list_mean_std_df, list_mean_tstat_df, list_median_iqr_df = [], [], []
    for i in np.arange(len(loss_fns)):
        file_path = folder_path.joinpath(f'loss_fn={loss_fns[i]}/score_fn={score_fn}/tau={tau}/performance/descriptive_stats/mean_std.csv')
        df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True) 
        try:
            df = df[df['strategy'] == use_strategy]
        except:
            pass
        list_mean_std_df.append(df)
        
        file_path = folder_path.joinpath(f'loss_fn={loss_fns[i]}/score_fn={score_fn}/tau={tau}/performance/descriptive_stats/mean_tstat.csv')
        df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True) 
        try:
            df = df[df['strategy'] == use_strategy]
        except:
            pass
        list_mean_tstat_df.append(df)
        
        file_path = folder_path.joinpath(f'loss_fn={loss_fns[i]}/score_fn={score_fn}/tau={tau}/performance/descriptive_stats/median_iqr.csv')
        df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True) 
        try:
            df = df[df['strategy'] == use_strategy]
        except:
            pass
        list_median_iqr_df.append(df)

    # display(list_mean_std_df[0].head() )
    # display(list_mean_std_df[1].head() )

    # Create a folder for output data
    out_dir = folder_path.joinpath('descriptive_stats')
    if not os.path.exists( out_dir ):
        # Create the directory if it does not exist .
        os.makedirs( out_dir )
    
    # Report the means (standard deviations) of a performance metric across loss functions
    n_rows = list_mean_std_df[0].shape[0]
    mean_std_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(len(loss_fns) + 3):
            if j < 3:
                mean_std_list[i].append(list_mean_std_df[0].values[i, j])
            else:
                mean_std_list[i].append(list_mean_std_df[j-3][perf_metric].values[i])

    mean_std_df = pd.DataFrame(mean_std_list, columns = list_mean_std_df[0].columns.tolist()[0:3] +  loss_fns)
    mean_std_df.to_csv(os.path.join(out_dir, f'mean_std_{score_fn}_perf_metric_{perf_metric}_tau={tau}.csv'), index = False, header = True)
    
    # Report the means (t-statistics) of a performance metric across loss functions
    n_rows = list_mean_tstat_df[0].shape[0]
    mean_tstat_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(len(loss_fns) + 3):
            if j < 3:
                mean_tstat_list[i].append(list_mean_tstat_df[0].values[i, j])
            else:
                mean_tstat_list[i].append(list_mean_tstat_df[j-3][perf_metric].values[i])

    mean_tstat_df = pd.DataFrame(mean_tstat_list, columns = list_mean_tstat_df[0].columns.tolist()[0:3] +  loss_fns)
    mean_tstat_df.to_csv(os.path.join(out_dir, f'mean_tstat_{score_fn}_perf_metric_{perf_metric}_tau={tau}.csv'), index = False, header = True)

    # Report the medians (interquartile ranges) of a performance metric across loss functions
    n_rows = list_median_iqr_df[0].shape[0]
    median_iqr_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(len(loss_fns) + 3):
            if j < 3:
                median_iqr_list[i].append(list_median_iqr_df[0].values[i, j])
            else:
                median_iqr_list[i].append(list_median_iqr_df[j-3][perf_metric].values[i])

    median_iqr_df = pd.DataFrame(median_iqr_list, columns = list_median_iqr_df[0].columns.tolist()[0:3] +  loss_fns)
    median_iqr_df.to_csv(os.path.join(out_dir, f'median_iqr_{score_fn}_perf_metric_{perf_metric}_tau={tau}.csv'), index = False, header = True)

    return True

def compare_predictors1 (use_predictors: List[str] = ['LGBM_SPY_all_vars', 'LGBM_SPY_all_vars_plus_patterns'], 
                                            tau: int = 1, # a forecast horizon
                                            loss_fn: str = 'Brier', use_strategy: str = 'fixed_trans_cost', perf_metric: str = 'cecpp'):
    ''' Compare the means (standard deviations),  means (t-statistics), and medians (IQRs) of a performance metric across 
        multiple sets of predictors, investment periods, and scoring functions for a given loss function
    INPUT
        use_predictors: a list of outputs from ML algorithms
        loss_fn: a loss function used for training (i.e., 'CE', 'Brier', 'Boost', 'As1', 'As2')
        use_strategy: a trading strategy (i.e., 'fixed_trans_cost' or 'variable_trans_cost')
        perf_metric: a metric used to evaluate the performance of a trading strategy (e.g., average_number_of_trades, percentage_of_winning_trades,	largest_raw_return,	
            smallest_raw_return,	ratio_win_loss, max_number_of_consecutive_winners, max_number_of_consecutive_losers, annualized_excess_return, annualized_standard_deviation,
            max_drawdown,	Schwager_gain-to-pain_ratio,	Calmar_ratio,	Sharpe_ratio,	Sortino_ratio,	cecpp,	mrar)
    OUTPUT
        dataframes of means (standard deviations), means (t-statistics), and medians (IQRs)
    '''
    # Read data as a list of dataframes
    folder_path = Path(f'../Results/')
    
    list_mean_std_df, list_mean_tstat_df, list_median_iqr_df = [], [], []
    for i in np.arange(len(use_predictors)):
        file_path = folder_path.joinpath(f'{use_predictors[i]}/loss_fn={loss_fn}/descriptive_stats/mean_std_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv')
        df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True) 
        list_mean_std_df.append(df)
        
        file_path = folder_path.joinpath(f'{use_predictors[i]}/loss_fn={loss_fn}/descriptive_stats/mean_tstat_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv')
        df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True) 
        list_mean_tstat_df.append(df)
        
        file_path = folder_path.joinpath(f'{use_predictors[i]}/loss_fn={loss_fn}/descriptive_stats/median_iqr_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv')
        df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True) 
        list_median_iqr_df.append(df)

    # display(list_mean_std_df[0].head() )
    # display(list_mean_std_df[1].head() )

    # Create a folder for output data
    algo_word = re.search(r'\w+(?=\_all)',  use_predictors[0], flags=re.IGNORECASE | re.VERBOSE).group()
    out_dir = folder_path.joinpath(f'descriptive_stats/{algo_word}')
    if not os.path.exists( out_dir ):
        # Create the directory if it does not exist .
        os.makedirs( out_dir )

    # Report the means (standard deviations) of a performance metric across sets of predictors
    n_rows, n_cols = list_mean_std_df[0].shape
    mean_std_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(n_cols):
            if j < 3:
                mean_std_list[i].append(list_mean_std_df[0].values[i, j])
            else:
                for k in np.arange(len(use_predictors)):
                    mean_std_list[i].append(list_mean_std_df[k].values[i, j])
    
    algo = re.search(r'^[^\_]+(?=\_)', use_predictors[0], flags=re.IGNORECASE | re.VERBOSE).group()
    list_columns = list_mean_std_df[0].columns.tolist()
    list_score_fns = [f'{list_columns[i]} ({use_predictors[j]})' for i in np.arange(3, len(list_columns)) for j in np.arange(len(use_predictors))]
    mean_std_df = pd.DataFrame(mean_std_list, columns = list_columns[0:3] +  list_score_fns)
    mean_std_df.to_csv(os.path.join(out_dir, f'mean_std_{loss_fn}_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv'), index = False, header = True)
    
    # Report the means (t-statistics) of a performance metric across sets of predictors
    n_rows, n_cols = list_mean_tstat_df[0].shape
    mean_tstat_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(n_cols):
            if j < 3:
                mean_tstat_list[i].append(list_mean_tstat_df[0].values[i, j])
            else:
                for k in np.arange(len(use_predictors)):
                    mean_tstat_list[i].append(list_mean_tstat_df[k].values[i, j])
    
    algo = re.search(r'^[^\_]+(?=\_)', use_predictors[0], flags=re.IGNORECASE | re.VERBOSE).group()
    list_columns = list_mean_tstat_df[0].columns.tolist()
    list_score_fns = [f'{list_columns[i]} ({use_predictors[j]})' for i in np.arange(3, len(list_columns)) for j in np.arange(len(use_predictors))]
    mean_tstat_df = pd.DataFrame(mean_tstat_list, columns = list_columns[0:3] +  list_score_fns)
    mean_tstat_df.to_csv(os.path.join(out_dir, f'mean_tstat_{loss_fn}_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv'), index = False, header = True)

    # Report the medians (IQRs) of a performance metric across sets of predictors
    n_rows, n_cols = list_median_iqr_df[0].shape
    median_iqr_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(n_cols):
            if j < 3:
                median_iqr_list[i].append(list_median_iqr_df[0].values[i, j])
            else:
                for k in np.arange(len(use_predictors)):
                    median_iqr_list[i].append(list_median_iqr_df[k].values[i, j])
    
    median_iqr_df = pd.DataFrame(median_iqr_list, columns = list_columns[0:3] +  list_score_fns)
    median_iqr_df.to_csv(os.path.join(out_dir, f'median_iqr_{loss_fn}_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv'), index = False, header = True)

    return True


def compare_predictors2 (  use_models: List[str] = ['LGBM_SPY_all_vars', 'LGBM_SPY_all_vars_plus_patterns'], 
                                            tau: int = 1, # a forecast horizon
                                            loss_fns: List[str] = ['CE', 'Brier', 'Boost', 'As1', 'As2'], 
                                            score_fn: str = 'AUC',
                                            use_strategy: str = 'fixed_trans_cost', 
                                            perf_metric: str = 'annualized_excess_return'):
    ''' Compare the means (standard deviations), means (t-statistics), and medians (IQRs) of a performance metric across multiple sets of predictors, 
        investment periods, and loss functions for a given scoring function.
    INPUT
        use_models: the output from ML models
        loss_fns: a list of loss functions used for training (i.e., 'CE', 'Brier', 'Boost', 'As1', 'As2')
        score_fn: a scoring function used for cross validation (i.e.,  'Accuracy', 'AUC', 'Gain_to_pain_ratio', 'Calmar_ratio', 'Sharpe_ratio', 'Sortino_ratio', 'CECPP')
        use_strategy: a trading strategy (i.e., 'fixed_trans_cost' or 'variable_trans_cost')
        perf_metric: a metric used to evaluate the performance of a trading strategy (e.g., average_number_of_trades, percentage_of_winning_trades,	largest_raw_return,	
            smallest_raw_return,	ratio_win_loss, max_number_of_consecutive_winners, max_number_of_consecutive_losers, annualized_excess_return, annualized_standard_deviation,
            max_drawdown,	Schwager_gain-to-pain_ratio,	Calmar_ratio,	Sharpe_ratio,	Sortino_ratio,	cecpp,	mrar)
    OUTPUT
        dataframes of means (standard deviations), means (t-statistics), and medians (IQRs)
    '''
    # Read data as a list of dataframes
    folder_path = Path(f'../Results/')
    
    list_mean_std_df, list_mean_tstat_df, list_median_iqr_df = [[] for _ in np.arange(len(loss_fns))], [[] for _ in np.arange(len(loss_fns))], [[] for _ in np.arange(len(loss_fns))]
    for i in np.arange(len(loss_fns)):
        for j in np.arange(len(use_models)):
            file_path = folder_path.joinpath(f'{use_models[j]}/loss_fn={loss_fns[i]}/descriptive_stats/mean_std_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv')
            df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True) 
            list_mean_std_df[i].append(df)
            
            file_path = folder_path.joinpath(f'{use_models[j]}/loss_fn={loss_fns[i]}/descriptive_stats/mean_tstat_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv')
            df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True) 
            list_mean_tstat_df[i].append(df)
            
            file_path = folder_path.joinpath(f'{use_models[j]}/loss_fn={loss_fns[i]}/descriptive_stats/median_iqr_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv')
            df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True) 
            list_median_iqr_df[i].append(df)

    # display(list_mean_std_df[0].head() )
    # display(list_mean_std_df[1].head() )

    # Create a folder for output data
    algo_word = re.search(r'\w+(?=\_all)',  use_models[0], flags=re.IGNORECASE | re.VERBOSE).group()
    out_dir = folder_path.joinpath(f'descriptive_stats/{algo_word}')
    if not os.path.exists( out_dir ):
        # Create the directory if it does not exist .
        os.makedirs( out_dir )

    if score_fn in ['Accuracy', 'AUC']:
        score_col = score_fn
    else:
        score_col = f'{score_fn}_{use_strategy}'
    
    # Report the means (standard deviations) of a performance metric across sets of predictors
    n_rows = list_mean_std_df[0][0].shape[0]
    mean_std_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(3):
            mean_std_list[i].append(list_mean_std_df[0][0].values[i, j])
        for k in np.arange(len(loss_fns)):
            for h in np.arange(len(use_models)):
                mean_std_list[i].append(list_mean_std_df[k][h][score_col].values[i])
    
    list_columns = list_mean_std_df[0][0].columns.tolist()
    list_loss_fns = [f'{loss_fns[i]} ({use_models[j]})' for i in np.arange(len(loss_fns)) for j in np.arange(len(use_models))]
    mean_std_df = pd.DataFrame(mean_std_list, columns = list_columns[0:3] +  list_loss_fns)
    mean_std_df.to_csv(os.path.join(out_dir, f'mean_std_{score_fn}_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv'), index = False, header = True)
    
    # Report the means (t-statistics) of a performance metric across sets of predictors
    n_rows = list_mean_tstat_df[0][0].shape[0]
    mean_tstat_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(3):
            mean_tstat_list[i].append(list_mean_tstat_df[0][0].values[i, j])
        for k in np.arange(len(loss_fns)):
            for h in np.arange(len(use_models)):
                mean_tstat_list[i].append(list_mean_tstat_df[k][h][score_col].values[i])
    
    list_columns = list_mean_tstat_df[0][0].columns.tolist()
    list_loss_fns = [f'{loss_fns[i]} ({use_models[j]})' for i in np.arange(len(loss_fns)) for j in np.arange(len(use_models))]
    mean_tstat_df = pd.DataFrame(mean_tstat_list, columns = list_columns[0:3] +  list_loss_fns)
    mean_tstat_df.to_csv(os.path.join(out_dir, f'mean_tstat_{score_fn}_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv'), index = False, header = True)

    # Report the medians (IQRs) of a performance metric across sets of predictors
    n_rows = list_median_iqr_df[0][0].shape[0]
    median_iqr_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(3):
            median_iqr_list[i].append(list_median_iqr_df[0][0].values[i, j])
        for k in np.arange(len(loss_fns)):
            for h in np.arange(len(use_models)):
                median_iqr_list[i].append(list_median_iqr_df[k][h][score_col].values[i])
    
    list_columns = list_median_iqr_df[0][0].columns.tolist()
    list_loss_fns = [f'{loss_fns[i]} ({use_models[j]})' for i in np.arange(len(loss_fns)) for j in np.arange(len(use_models))]
    median_iqr_df = pd.DataFrame(median_iqr_list, columns = list_columns[0:3] +  list_loss_fns)
    median_iqr_df.to_csv(os.path.join(out_dir, f'median_iqr_{score_fn}_{use_strategy}_perf_metric_{perf_metric}_tau={tau}.csv'), index = False, header = True)

    return True

def compare_perf_metrics (use_models: List[str] = ['LGBM_SPY_all_vars', 'LGBM_SPY_all_vars_plus_patterns'], 
                                            tau: int = 1, # a forecast horizon
                                            loss_fn: str = 'Brier',
                                            score_fn: str = 'CECPP_fixed_trans_cost'):
    ''' Compare the means (standard deviations), means (t-statistics), and medians (IQRs) of all the performance metrics across 
        multiple sets of predictors, investment periods, and amounts of transaction cost for a given loss function and scoring function.
    INPUT
        use_models: the output from ML models
        loss_fn: a loss function used for training (i.e., 'CE', 'Brier', 'Boost', 'As1', 'As2')
        score_fn: a scoring function used for cross validation (i.e.,  'Accuracy', 'AUC', 'Gain_to_pain_ratio', 'Calmar_ratio', 'Sharpe_ratio', 'Sortino_ratio', 'CECPP')
    OUTPUT
        dataframes of means (standard deviations), means (t-statistics), and medians (IQRs)
    '''
    # Read data as a list of dataframes
    folder_path = Path(f'../Results/')
    
    use_strategy = re.search(r'(fixed|variable)\w+', score_fn, flags=re.IGNORECASE | re.VERBOSE)
    columns_to_select = ['strategy', 'holding_per',	'trans_cost', 'percentage_of_winning_trades', 'ratio_win_loss', 'annualized_excess_return', \
                                                    'annualized_standard_deviation', 'Schwager_gain-to-pain_ratio', 'Sharpe_ratio', 'Sortino_ratio', 'cecpp', 'mrar']
    
    list_mean_std_df, list_mean_tstat_df, list_median_iqr_df = [], [], []
    for i in np.arange(len(use_models)):
        file_path = folder_path.joinpath(f'{use_models[i]}/loss_fn={loss_fn}/score_fn={score_fn}/tau={tau}/performance/descriptive_stats/mean_std.csv')
        df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True) 
        df = df[columns_to_select]
        if use_strategy:
            df = df[df['strategy'] == use_strategy.group()]
        list_mean_std_df.append(df)
        
        file_path = folder_path.joinpath(f'{use_models[i]}/loss_fn={loss_fn}/score_fn={score_fn}/tau={tau}/performance/descriptive_stats/mean_tstat.csv')
        df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True) 
        df = df[columns_to_select]
        if use_strategy:
            df = df[df['strategy'] == use_strategy.group()]
        list_mean_tstat_df.append(df)
        
        file_path = folder_path.joinpath(f'{use_models[i]}/loss_fn={loss_fn}/score_fn={score_fn}/tau={tau}/performance/descriptive_stats/median_iqr.csv')
        df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace=True) 
        df = df[columns_to_select]
        if use_strategy:
            df = df[df['strategy'] == use_strategy.group()]
        list_median_iqr_df.append(df)

    # display(list_mean_std_df[0].head() )
    # display(list_mean_std_df[1].head() )

    # Create a folder for output data
    algo_word = re.search(r'\w+(?=\_all)',  use_models[0], flags=re.IGNORECASE | re.VERBOSE).group()
    out_dir = folder_path.joinpath(f'descriptive_stats/{algo_word}')
    if not os.path.exists( out_dir ):
        # Create the directory if it does not exist .
        os.makedirs( out_dir )
    
    # Report the means (standard deviations) of all the performance metrics across sets of predictors
    n_rows, n_cols = list_mean_std_df[0].shape
    mean_std_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(3):
            mean_std_list[i].append(list_mean_std_df[0].values[i, j])
        for h in np.arange(3, n_cols):
            for k in np.arange(len(use_models)):
                mean_std_list[i].append(list_mean_std_df[k].values[i, h])
    
    list_columns = list_mean_std_df[0].columns.tolist()
    list_perf_metrics = [f'{list_columns[i]} ({use_models[j]})' for i in np.arange(3, n_cols) for j in np.arange(len(use_models))]
    mean_std_df = pd.DataFrame(mean_std_list, columns = list_columns[0:3] +  list_perf_metrics)
    mean_std_df.to_csv(os.path.join(out_dir, f'mean_std_{loss_fn}_{score_fn}_all_perf_metrics_tau={tau}.csv'), index = False, header = True)
    
    # Report the means (t-statistics) of all the performance metrics across sets of predictors
    n_rows, n_cols = list_mean_tstat_df[0].shape
    mean_tstat_list = [ [ ] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(3):
            mean_tstat_list[i].append(list_mean_tstat_df[0].values[i, j])
        for h in np.arange(3, n_cols):
            for k in np.arange(len(use_models)):
                mean_tstat_list[i].append(list_mean_tstat_df[k].values[i, h])
    
    list_columns = list_mean_tstat_df[0].columns.tolist()
    list_perf_metrics = [f'{list_columns[i]} ({use_models[j]})' for i in np.arange(3, n_cols) for j in np.arange(len(use_models))]
    mean_tstat_df = pd.DataFrame(mean_tstat_list, columns = list_columns[0:3] +  list_perf_metrics)
    mean_tstat_df.to_csv(os.path.join(out_dir, f'mean_tstat_{loss_fn}_{score_fn}_all_perf_metrics_tau={tau}.csv'), index = False, header = True)

    # Report the medians (IQRs) of all the performance metrics across sets of predictors
    median_iqr_list = [ [] for _ in np.arange(n_rows) ]
    for i in np.arange(n_rows):
        for j in np.arange(3):
            median_iqr_list[i].append(list_median_iqr_df[0].values[i, j])
        for h in np.arange(3, n_cols):
            for k in np.arange(len(use_models)):
                median_iqr_list[i].append(list_median_iqr_df[k].values[i, h])
    
    median_iqr_df = pd.DataFrame(median_iqr_list, columns = list_columns[0:3] +  list_perf_metrics)
    median_iqr_df.to_csv(os.path.join(out_dir, f'median_iqr_{loss_fn}_{score_fn}_all_perf_metrics_tau={tau}.csv'), index = False, header = True)
    return True


if __name__ == "__main__":
    startTime = time.time()

    # Define a list of outputs from ML algorithms employed
    algos = ['RF_BTC_all_vars', 'RF_BTC_all_vars_plus_patterns']
    # algos = ['XGB_SPY_all_vars', 'XGB_SPY_all_vars_plus_patterns']
    # algos = ['LGBM_SPY_all_vars', 'LGBM_SPY_all_vars_plus_patterns']
    # algos = ['RF_RW_all_vars', 'RF_RW_all_vars_plus_patterns']

    # Define a list of loss functions used to train a ML model
    # loss_fns = ['CE', 'Brier', 'Boost', 'As1', 'As2']
    loss_fns = ['CE']
    
    # Define a list of forecast horizons
    taus = [1, 2]

    # Define a list of score functions used to cross validate a ML algorithm
    # score_fns = ['Accuracy', 'AUC',  'Gain_to_pain_ratio_fixed_trans_cost',  'Gain_to_pain_ratio_variable_trans_cost', 'Calmar_ratio_fixed_trans_cost', \
    #                     'Calmar_ratio_variable_trans_cost', 'Sharpe_ratio_fixed_trans_cost', 'Sharpe_ratio_variable_trans_cost', 'Sortino_ratio_fixed_trans_cost', \
    #                     'Sortino_ratio_variable_trans_cost', 'CECPP_fixed_trans_cost', 'CECPP_variable_trans_cost']   
    score_fns_fixed_trans_cost = ['Accuracy', 'AUC',  'Gain_to_pain_ratio_fixed_trans_cost', 'Calmar_ratio_fixed_trans_cost', \
                                                        'Sharpe_ratio_fixed_trans_cost',  'Sortino_ratio_fixed_trans_cost',  'CECPP_fixed_trans_cost']   
    # score_fns_variable_trans_cost = ['Accuracy', 'AUC',  'Gain_to_pain_ratio_variable_trans_cost', 'Calmar_ratio_variable_trans_cost', \
    #                                                         'Sharpe_ratio_variable_trans_cost', 'Sortino_ratio_variable_trans_cost', 'CECPP_variable_trans_cost'] 

    # Define a list of performance metrics
    perf_metrics = ['average_number_of_trades', 'percentage_of_winning_trades',	'largest_raw_return', 'smallest_raw_return', 'ratio_win_loss', \
                                'max_number_of_consecutive_winners', 'max_number_of_consecutive_losers', 'annualized_excess_return', 'annualized_standard_deviation', \
                                'max_drawdown',	'Schwager_gain-to-pain_ratio', 'Calmar_ratio', 'Sharpe_ratio', 'Sortino_ratio',	'cecpp', 'mrar']

    # Calculate the means, standard deviations,  t-statistics, medians, and IQRs of the performance metrics over multiple investment periods
    for algo in algos:
        for loss_fn in loss_fns:
            for score_fn in score_fns_fixed_trans_cost:
                for tau in taus:
                    summary_stats(use_model = algo, tau = tau, loss_fn = loss_fn, score_fn = score_fn)
    
    # Compare the means (standard deviations), means (t-statistics), and medians (IQRs) of a performance metric over multiple investment periods across various score functions for a given loss function
    for algo in algos:
        for loss_fn in loss_fns:
            for perf_metric in perf_metrics:
                # for score_fn_ls in [score_fns_fixed_trans_cost,  score_fns_variable_trans_cost]:
                for score_fn_ls in [score_fns_fixed_trans_cost]:
                    for tau in taus:
                        compare_score_fns(use_model = algo, loss_fn = loss_fn, tau = tau, score_fns = score_fn_ls, perf_metric = perf_metric)

    # # Compare the means (standard deviations), means (t-statistics), and medians (IQRs) of a performance metric over multiple investment periods and loss functions for a given scoring function
    for algo in algos:
        for score_fn in score_fns_fixed_trans_cost:
            for perf_metric in perf_metrics:
                for tau in taus:
                    print(f'algo = {algo}, score_fn = {score_fn}, perf_metric = {perf_metric}, tau = {tau}')
                    compare_loss_fns(use_model = algo, loss_fns = loss_fns, tau = tau, score_fn = score_fn, perf_metric = perf_metric)

    # Compare the means (standard deviations), means (t-statistics), and medians (IQRs) of a performance metric across multiple sets of predictors, investment periods, and scoring functions 
    # for a given loss function
    for loss_fn in loss_fns:
        for perf_metric in perf_metrics:
            for tau in taus:
                compare_predictors1(use_predictors = algos, loss_fn = loss_fn, tau = tau, perf_metric = perf_metric)

    # Compare the means (standard deviations), means (t-statistics), and medians (IQRs) of a performance metric across multiple sets of predictors, investment periods, and loss functions 
    # for a given scoring function
    for score_fn in ['Accuracy', 'AUC', 'Gain_to_pain_ratio', 'Calmar_ratio', 'Sharpe_ratio', 'Sortino_ratio', 'CECPP']:
        for perf_metric in perf_metrics:
            for use_strategy in ['fixed_trans_cost']:
                for tau in taus:
                    compare_predictors2(use_models = algos, loss_fns = loss_fns, tau = tau, score_fn = score_fn, use_strategy = use_strategy, perf_metric = perf_metric)
    
    # Compare the means (standard deviations), means (t-statistics), and medians (IQRs) of all the performance metrics across multiple sets of predictors, investment periods, 
    # and amounts of transaction cost for a given loss function and scoring function.
    for loss_fn in loss_fns:
        for score_fn in score_fns_fixed_trans_cost:
            for tau in taus:
                compare_perf_metrics (use_models = algos, tau = tau, loss_fn = loss_fn, score_fn = score_fn)
    
    print( 'The script took {} second !'.format(time.time() - startTime) )