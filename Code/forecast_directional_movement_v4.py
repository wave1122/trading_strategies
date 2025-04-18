# =================================================== Compute Out-of-Sample Classification Metrics ========================================================= #
import os
import sys

# add the path to the working directory
try:
    path="/home/bachu/Research/Dropbox/Codes/AMD-ThreadRipper-3990X-1/ForecastStocks/Jupyter_notebooks"
    sys.path.append(path)
except:
    try:
        path="../"
        sys.path.append(path)
    except Exception as e:
        print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        
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

import logging

##### Import algorithms
import my_classification_algorithms_optuna_v4 as optuna_cl
import custom_losses_v1 as closs

##### Calculate rolling-window 'tau'-steps ahead forecasts
def rolling_forecast(   df: pd.DataFrame, # a pandas dataframe
                                    price_RF: pd.DataFrame, # a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags
                                    tau: int, # a forecast horizon
                                    wsize: int, # the size of a rolling window
                                    ylag: int, # the number of lagged outcome variables
                                    use_model = 'LGBM', 
                                    objective = closs.update_As1_lgbm, # a LGBM objective function
                                    objective_name = 'As1', # the name of the objective function
                                    scoring_fn = 'AUC',  # a scoring function used to cross validate a LightGBM model 
                                                      # This argument can takes: ['Accuracy', 'Average_Precision', 'Precision', 'F1_score', 'AUC', 'Cross_entropy', 'As1_score', 'As2_score', 'Boost_score', 'Brier_score', 
                                                      # 'Gain_to_pain_ratio_fixed_trans_cost', 'Gain_to_pain_ratio_variable_trans_cost', 'Calmar_ratio_fixed_trans_cost', 'Calmar_ratio_variable_trans_cost', 
                                                      # 'Sharpe_ratio_fixed_trans_cost', 'Sharpe_ratio_variable_trans_cost', 'Sortino_ratio_fixed_trans_cost', 'Sortino_ratio_variable_trans_cost', 
                                                      # 'CECPP_fixed_trans_cost', 'CECPP_variable_trans_cost']
                                    use_custom_loss = True, # whether or not to use a custom loss function for LGBM
                                    init_wealth = 1000, # an initial wealth
                                    fixed_trans_cost = 10, # the dollar amount of fixed transaction cost 
                                    variable_trans_cost = 0.005, # an amount of transaction cost as the percentage of stock price
                                    n_trials = 100, # the number of trials set for Optuna
                                    n_jobs = -1, # the number of workers for multiprocessing
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Implement the rolling-window forecast strategy
        INPUT
            df: a pandas dataframe beginning with columns [date, direction,	return,	log_return]
            price_RF: a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags
            tau: forecast horizon
            wsize: window size
            ylag: AR lag of arithmetic returns
            use_model: to use a ML method
            objective: an objective function required by LightGBM
            scoring_fn: a score function used to cross evaluate forecasts
            use_custom_loss: whether or not to use a custom loss function for LGBM
            init_wealth: an initial wealth
            fixed_trans_cost: the dollar amount of fixed transaction cost 
            variable_trans_cost: an amount of transaction cost as the percentage of stock price
            n_trials: the number of trials required by Optuna

       OUTPUT
            dataframes (probability forecasts, scores, optimal hyperparameters, importance scores, SHAP values)
    """
    T1 = df.shape[0] # get the number of time periods
    assert (T1 > wsize+tau), "The rolling-window size must be smaller than the number of time periods!"

    # define a logger
    logging.basicConfig(filename = f'./Log/log_{objective_name}.txt', format="%(asctime)s %(message)s", filemode="w")
    logger = logging.getLogger(objective_name)
    logger.setLevel(logging.INFO)

    df = df.copy() # create a deep copy of the original dataframe

    # get data for the dependant variable and predictors
    if ylag > 0:
        for i in np.arange(0, ylag+1):
            df[f'ret_lag{i}'] = df['return'].shift(i)
        df.dropna(inplace = True)
    # print( df.iloc[0, 0] )

    R_df = df['direction']
    X_df = df.loc[:, 'log_volume':]
    X = np.array(X_df.values, dtype='float64')
    dim = X.shape[1]

    # time the execution
    start = time.time()

    start_dates, trans_dates, end_dates, actuals, prob_forecasts = [], [], [], [], []
    list_mean_scores_values, list_opt_params_vlues, list_feat_importances_values, list_SHAP_values = [], [], [], []
    for s in np.arange(0, T1-wsize-tau-ylag, tau):
        logger.info(f'{use_model} cross-validated using the scoring function = {scoring_fn} -- Rolling window #: {s}')
        start_dates.append(df.index[s])
        trans_dates.append(df.index[s+wsize])
        end_dates.append(df.index[s+wsize+tau])

        # estimate a classification model, and make a 'tau'-steps ahead forecast
        if use_model == 'LGBM':
            prob_tau, scores_mean, opt_params, features_importances_df, ave_SHAP_vlues_df = optuna_cl.LGBMf ( R_df.iloc[s:(s+wsize+1)],
                                                                                                                                                                                        X[s:(s+wsize+1), :].reshape(-1, dim),
                                                                                                                                                                                        data_df = price_RF,
                                                                                                                                                                                        feat_names = X_df.columns,
                                                                                                                                                                                        tau = tau,
                                                                                                                                                                                        objective = objective,
                                                                                                                                                                                        scoring_fn = scoring_fn,
                                                                                                                                                                                        use_custom_loss = use_custom_loss,
                                                                                                                                                                                        init_wealth = init_wealth,
                                                                                                                                                                                        fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                        variable_trans_cost = variable_trans_cost,
                                                                                                                                                                                        n_trials = n_trials,
                                                                                                                                                                                        n_jobs = n_jobs,
                                                                                                                                                                                    )
        elif use_model == 'RF':
            prob_tau, scores_mean, opt_params, features_importances_df, ave_SHAP_vlues_df  = optuna_cl.RFf(    R_df.iloc[s:(s+wsize+1)],
                                                                                                                                                                                    X[s:(s+wsize+1), :].reshape(-1, dim),
                                                                                                                                                                                    data_df = price_RF,
                                                                                                                                                                                    feat_names = X_df.columns,
                                                                                                                                                                                    tau = tau,
                                                                                                                                                                                    scoring_fn = scoring_fn,
                                                                                                                                                                                    init_wealth = init_wealth,
                                                                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                    variable_trans_cost = variable_trans_cost,
                                                                                                                                                                                    n_trials = n_trials,
                                                                                                                                                                                    n_jobs = n_jobs,
                                                                                                                                                                                )
        elif use_model == 'XGB':
            prob_tau, scores_mean, opt_params, features_importances_df, ave_SHAP_vlues_df  = optuna_cl.XGBf(R_df.iloc[s:(s+wsize+1)],
                                                                                                                                                                                    X[s:(s+wsize+1), :].reshape(-1, dim),
                                                                                                                                                                                    data_df = price_RF,
                                                                                                                                                                                    feat_names = X_df.columns,
                                                                                                                                                                                    tau = tau,
                                                                                                                                                                                    scoring_fn = scoring_fn,
                                                                                                                                                                                    init_wealth = init_wealth,
                                                                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                    variable_trans_cost = variable_trans_cost,
                                                                                                                                                                                    n_trials = n_trials,
                                                                                                                                                                                    n_jobs = n_jobs,
                                                                                                                                                                                    )
        
        else:
            print(f'Model {use_model} does not exist!')
            sys.exit()

        # save mean scores to list
        list_mean_scores_keys = [k for k in scores_mean.keys()]
        list_mean_scores_values.append( list( scores_mean.values() ) )

        # save optimal hyperparameters to list
        list_opt_params_keys = [k for k in opt_params.keys()]
        list_opt_params_vlues.append( list( opt_params.values() ) )

        # save feature importance scores to list
        list_feat_importances_values.append( features_importances_df.importance_score.tolist() )

        # save average SHAP values to list
        list_SHAP_values.append( ave_SHAP_vlues_df.average_SHAP_value.tolist() )

        actuals.append(R_df.iat[s+wsize+tau]) # actual returns
        prob_forecasts.append(prob_tau)

    # save forecasts to a dataframe
    forecasts_df = pd.DataFrame({'start_date': start_dates, 'trans_date': trans_dates, 'end_date': end_dates, 'actual': actuals, 'proba_forecast': prob_forecasts})

    # save mean scores to a dataframe
    list_mean_scores_values = np.array(list_mean_scores_values)
    scores_df = pd.DataFrame({list_mean_scores_keys[i]: list_mean_scores_values[:, i] for i in np.arange( len(list_mean_scores_keys) )})
    scores_df.insert(loc=0, column='start_date', value=start_dates)
    scores_df.insert(loc=1, column='trans_date', value=trans_dates)
    scores_df.insert(loc=2, column='end_date', value=end_dates)

    # save optimal hyperparameters to list
    list_opt_params_vlues = np.array(list_opt_params_vlues)
    opt_params_df = pd.DataFrame({list_opt_params_keys[i]: list_opt_params_vlues[:, i] for i in np.arange( len(list_opt_params_keys) )})
    opt_params_df.insert(loc=0, column='start_date', value=start_dates)
    opt_params_df.insert(loc=1, column='trans_date', value=trans_dates)
    opt_params_df.insert(loc=2, column='end_date', value=end_dates)

    # save feature importance scores to a dataframe
    list_feat_importances_values = np.array(list_feat_importances_values)
    importances_df = pd.DataFrame({features_importances_df.iloc[i]['feature']: list_feat_importances_values[:, i] for i in np.arange( list_feat_importances_values.shape[1] )})
    importances_df.insert(loc=0, column='start_date', value=start_dates)
    importances_df.insert(loc=1, column='trans_date', value=trans_dates)
    importances_df.insert(loc=2, column='end_date', value=end_dates)

    # save everage SHAP values to a dataframe
    list_SHAP_values = np.array(list_SHAP_values)
    SHAP_df = pd.DataFrame({ave_SHAP_vlues_df.iloc[i]['feature']: list_SHAP_values[:, i] for i in np.arange( list_SHAP_values.shape[1] )})
    SHAP_df.insert(loc=0, column='start_date', value=start_dates)
    SHAP_df.insert(loc=1, column='trans_date', value=trans_dates)
    SHAP_df.insert(loc=2, column='end_date', value=end_dates)
    
    del df # delete this copy of the dataframe
    gc.collect()

    end = time.time()
    print( 'Completed in: %s sec'%(end - start) )

    return forecasts_df, scores_df, opt_params_df, importances_df, SHAP_df

