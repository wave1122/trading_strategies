# ========================================================== Test the Trading Strategies ============================================================ #
import pandas as pd
import numpy as np
rng = np.random.RandomState(seed=35)
import time
import os
from math import floor, ceil

from dask.distributed import Client, LocalCluster
import joblib
import multiprocessing
# import dask
# import distributed
# dask.config.set({"distributed.comm.timeouts.tcp": "100000s", "distributed.scheduler.allowed-failures": 999})
# num_cores = multiprocessing.cpu_count()
num_cores = 30

# ##### Set the current working directory
# path="e:/Copy/SCRIPTS/Forecast_Stocks/Jupyter_notebooks/"
# os.chdir(path)

import trading_strategies as tt
from performance_summary import perf_summary_report, perf_summary_report_all_periods

def perf_report(   invest_window = 100, # an investment period
                            tau = 2, # a forecast horizon
                            fixed_trans_cost = 10, # an amount of fixed transaction cost
                            variable_trans_cost = 0.005, # a variable transaction cost as the percentage of stock price
                            use_strategy = 'fixed_trans_cost', # a trading strategy to use
                            use_model = 'LGBM_SPY_all_vars', 
                            loss_fn_name = 'CE', 
                            score_fn = 'AUC', 
                            wsize = 1000, # a rolling window size used to train a ML model
                            n_trials = 30, 
                            init_wealth = 1000, 
                            fixed_trans_cost_train = 10, 
                            variable_trans_cost_train = 0.005
                        ):
    ''' Report all the performance metrics of trading strategies using a forecasting method
    INPUT
        invest_window: an investment period (in days)
        tau: a forecast horizon
        fixed_trans_cost: an amount of fixed transaction cost used for the out-of-sample evaluation of a trading strategy
        variable_trans_cost: a variable transaction cost as the percentage of stock price used for the out-of-sample evaluation of a trading strategy
        use_strategy: a trading strategy to use
        use_model: the name of a ML method used to make predictions 
        loss_fn_name: the name of a loss function used to train the ML model
        score_fn: the name of a score function
        wsize: a rolling window size used to train a ML model
        n_trials: the number of trials set for Optuna to cross-validate the ML model
        init_wealth: an initial endowment 
        fixed_trans_cost_train: an amount of fixed transaction cost used to train a ML model
        variable_trans_cost_train: a variable transaction cost used to train a ML model
    OUTPUT
        a dataframe containing all performance metrics across many different holding periods
    '''
    # Read forecasts of stock returns
    forecast_df = pd.read_csv(f'../Results/{use_model}/loss_fn={loss_fn_name}/score_fn={score_fn}/tau={tau}/forecast_wsize_{wsize}_n_trials_{n_trials}'\
                                                f'_init_wealth_{init_wealth}_fixed_trans_cost_{fixed_trans_cost_train}_variable_trans_cost_{variable_trans_cost_train}.csv', \
                                                encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace = True, parse_dates = ['start_date', 'trans_date',	'end_date'])                                                                                                                                                                                    
    # display(forecast_df.head() )

    # Read stock prices and risk-free interest rate
    price_df = pd.read_csv('../Data/SPY.csv', encoding='utf-8', sep = ',', low_memory = False, header = 0, skiprows = 0, skipinitialspace = True, usecols = ['date', 'Close', 'RF'], parse_dates = ['date'])
    price_df.rename(columns = {'Close': 'price'}, inplace = True)
    # display(price_df.head() )

    perf_metrics_all_periods_df = perf_summary_report_all_periods(forecast_df, price_df, init_wealth = init_wealth, fixed_trans_cost = fixed_trans_cost, variable_trans_cost = variable_trans_cost, \
                                                                                                                                                                                    invest_window = floor(invest_window/tau), freq = tau, use_strategy = use_strategy)

    out_dir = f'../Results/{use_model}/loss_fn={loss_fn_name}/score_fn={score_fn}/tau={tau}/performance/'

    if not os.path.exists( out_dir ):
        # Create the directory if it does not exist .
        os.makedirs( out_dir )
    
    if use_strategy == 'fixed_trans_cost':
            out_file =  os.path.join(out_dir, f'performance_hper_{invest_window}_init_wealth_{init_wealth}_fixed_trans_cost_{fixed_trans_cost}.csv')
    elif use_strategy ==  'variable_trans_cost':
            out_file =  os.path.join(out_dir, f'performance_hper_{invest_window}_init_wealth_{init_wealth}_variable_trans_cost_{variable_trans_cost}.csv')
    else:
            raise Exception(f'Strategy \'{use_strategy}\' does not exist!')
            sys.exit()

    perf_metrics_all_periods_df.to_csv(out_file, index = False, header = True)

    # return perf_metrics_all_periods_df
    return True

if __name__ == "__main__":
    startTime = time.time()

    use_model = 'LGBM_SPY_all_vars'
    # Define a list of loss functions used to train a ML model
    loss_fns = ['CE', 'Brier', 'Boost', 'As1', 'As2']
    # loss_fns = ['CE']

    n_trials = 30

    # Define lists of transaction costs used to trade
    fixed_trans_costs = [0.05, 0.1, 0.5, 1.0, 5.0]
    # variable_trans_costs = [0.001, 0.002, 0.005, 0.01, 0.05]
    # variable_trans_costs = [0.0005]

    # Define a list of score functions used to cross validate a ML algorithm
    # score_fns_fixed_trans_cost = ['Accuracy', 'AUC',  'Gain_to_pain_ratio_fixed_trans_cost', 'Calmar_ratio_fixed_trans_cost', \
    #                                                     'Sharpe_ratio_fixed_trans_cost',  'Sortino_ratio_fixed_trans_cost',  'CECPP_fixed_trans_cost']   
    score_fns_fixed_trans_cost = ['Accuracy', 'AUC',  'Gain_to_pain_ratio_fixed_trans_cost', 'Sharpe_ratio_fixed_trans_cost', 'CECPP_fixed_trans_cost']
    
    # score_fns_variable_trans_cost = ['Accuracy', 'AUC',  'Gain_to_pain_ratio_variable_trans_cost', 'Calmar_ratio_variable_trans_cost', \
    #                                                         'Sharpe_ratio_variable_trans_cost', 'Sortino_ratio_variable_trans_cost', 'CECPP_variable_trans_cost']  

    # Define a list of holding periods
    invest_windows = [100, 200]
    
    # Define a list of forecast horizons
    taus = [2, 4, 6, 8, 10, 12]

    try:
        # client = Client('tcp://localhost:8786', timeout='2s')
        cluster = LocalCluster(n_workers=num_cores, processes=True, memory_limit='auto', threads_per_worker=1, scheduler_port=8786, dashboard_address='localhost:8787')
        client = Client(cluster)
    except OSError:
        client.close()
        cluster.close()
        time.sleep(20)
        cluster = LocalCluster(n_workers=num_cores, processes=True, memory_limit='auto', threads_per_worker=1, scheduler_port=8786, dashboard_address='localhost:8787')
        client = Client(cluster)
    print(client)
    
    use_strategy = 'fixed_trans_cost'
    with joblib.parallel_backend('dask'):
        job_run = joblib.Parallel(verbose=20) (joblib.delayed(perf_report)(invest_window = invest_window, # an investment period
                                                                                                                    tau = tau, # a forecast horizon
                                                                                                                    fixed_trans_cost = fixed_trans_cost, # an amount of fixed transaction cost
                                                                                                                    variable_trans_cost = 0, # a variable transaction cost as the percentage of stock price
                                                                                                                    use_strategy = use_strategy, # a trading strategy to use
                                                                                                                    use_model = use_model, 
                                                                                                                    loss_fn_name = loss_fn, 
                                                                                                                    score_fn = score_fn, 
                                                                                                                    wsize = 1000, # a rolling window size used to train a ML model
                                                                                                                    n_trials = n_trials, 
                                                                                                                    init_wealth = 1000, 
                                                                                                                    fixed_trans_cost_train = 10, 
                                                                                                                    variable_trans_cost_train = 0.005) \
                                                                                                                    for invest_window in invest_windows
                                                                                                                        for tau in taus
                                                                                                                            for loss_fn in loss_fns
                                                                                                                                for fixed_trans_cost in fixed_trans_costs 
                                                                                                                                    for score_fn in score_fns_fixed_trans_cost) 
    # time.sleep(60)

    # use_strategy = 'variable_trans_cost'
    # with joblib.parallel_backend('dask'):
    #     job_run = joblib.Parallel(verbose=20) (joblib.delayed(perf_report)(invest_window = invest_window, # a holding period
    #                                                                                                                 fixed_trans_cost = 0, # an amount of fixed transaction cost
    #                                                                                                                 variable_trans_cost = variable_trans_cost, # a variable transaction cost as the percentage of stock price
    #                                                                                                                 use_strategy = use_strategy, # a trading strategy to use
    #                                                                                                                 use_model = use_model,
    #                                                                                                                 loss_fn_name = loss_fn, 
    #                                                                                                                 score_fn = score_fn, 
    #                                                                                                                 wsize = 1000, # a rolling window size used to train a ML model
    #                                                                                                                 n_trials = n_trials, 
    #                                                                                                                 init_wealth = 1000, 
    #                                                                                                                 fixed_trans_cost_train = 10, 
    #                                                                                                                 variable_trans_cost_train = 0.005) \
    #                                                                                                                 for invest_window in invest_windows
    #                                                                                                                     for loss_fn in loss_fns
    #                                                                                                                         for variable_trans_cost in variable_trans_costs 
    #                                                                                                                             for score_fn in score_fns_variable_trans_cost) 
    
    # time.sleep(60)

    client.close()
    cluster.close()
    
    print( 'The script took {} second !'.format(time.time() - startTime) )