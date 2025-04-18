##### Set the current working directory
import sys
try:
    path="/home/bachu/Research/Dropbox/Codes/AMD-ThreadRipper-3990X-1/ForecastStocks/Jupyter_notebooks"
    sys.path.append(path)
except:
    try:
        path="../"
        sys.path.append(path)
    except Exception as e:
        print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

import pandas as pd
import numpy as np
import os
import time
from varname import nameof

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# suppress warnings
warnings.filterwarnings('ignore')

import my_classification_algorithms_optuna_v4 as optuna_cl
import forecast_directional_movement_v3 as forecast
import custom_losses_v1 as closs

if __name__ == "__main__":
    data_df = pd.read_csv('../Data/SPY_all_vars.csv', encoding='utf-8', sep = ',', low_memory=False, header = 0, skiprows = 0, skipinitialspace=True, parse_dates = ['date'])
    
    tau = 2 # this is a forecast horizon that can take value in [2, 4, 6, 8, 10, 12]
    
    # Select the columns: 'date', 'price', and 'RF', then shift the last two columns forward 'tau' lags
    price_RF = data_df[['date', 'price', 'RF']]
    price_RF[['price', 'RF']] = price_RF[['price', 'RF']].shift(periods = -tau)
    
    data_df.set_index('date', inplace = True)
    # display( data_df.head() )

    use_data = 'SPY_all_vars'

    use_model = 'RF'

    use_custom_loss = False # whether or not to use a custom loss function for LGBM

    # Define objective/loss functions for LGBM
    objective = closs.update_Brier_lgbm
    loss_fn_name = 'Brier'

    if use_model == 'LGBM':
        if use_custom_loss:
            loss_fn_name = loss_fn_name
        else:
            loss_fn_name = 'CE'
        n_trials = 30 # set the number of trials for Optuna
    elif use_model in ['RF', 'XGB']:
        loss_fn_name = 'CE'
        n_trials = 30 # set the number of trials for Optuna
    else:
        print(f'Model {use_model} does not exist!')
        sys.exit()

    # List scoring functions used to rank models during the cross validation. 
    # This argument can takes: ['Accuracy', 'Average_Precision', 'Precision', 'F1_score', 'AUC', 'Cross_entropy', 'As1_score', 'As2_score', 'Boost_score', 'Brier_score', 
                                                # 'Gain_to_pain_ratio_fixed_trans_cost', 'Gain_to_pain_ratio_variable_trans_cost', 'Calmar_ratio_fixed_trans_cost', 'Calmar_ratio_variable_trans_cost', 
                                                # 'Sharpe_ratio_fixed_trans_cost', 'Sharpe_ratio_variable_trans_cost', 'Sortino_ratio_fixed_trans_cost', 'Sortino_ratio_variable_trans_cost', 
                                                # 'CECPP_fixed_trans_cost', 'CECPP_variable_trans_cost']
                                                
    # scoring_fns = ['Accuracy', 'AUC', 'Gain_to_pain_ratio_fixed_trans_cost', 'Gain_to_pain_ratio_variable_trans_cost', 'Calmar_ratio_fixed_trans_cost', 'Calmar_ratio_variable_trans_cost', 
                            # 'Sharpe_ratio_fixed_trans_cost', 'Sharpe_ratio_variable_trans_cost', 'Sortino_ratio_fixed_trans_cost', 'Sortino_ratio_variable_trans_cost', 
                            # 'CECPP_fixed_trans_cost', 'CECPP_variable_trans_cost']   

    scoring_fns = ['Accuracy', 'AUC', 'Calmar_ratio_fixed_trans_cost', 'Sortino_ratio_fixed_trans_cost']   

    wsize = 1000 # set the rolling window size
    init_wealth = 1000 # an initial wealth
    fixed_trans_cost = 10 # the dollar amount of fixed transaction cost 
    variable_trans_cost = 0.005 # an amount of transaction cost as the percentage of stock price

    # Start timing the main loop
    start = time.time()

    for scoring_fn in scoring_fns:
        out_dir = f'./Results/{use_model}_{use_data}/loss_fn={loss_fn_name}/score_fn={scoring_fn}/tau={tau}/'
        if not os.path.exists( out_dir ):
        # Create the directory if it does not exist .
            os.makedirs( out_dir )

        forecasts_df, scores_df, opt_params_df, importances_df, SHAP_df = forecast.rolling_forecast(data_df,
                                                                                                                                                                price_RF = price_RF, # a dataframe with columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags
                                                                                                                                                                tau = tau, # a forecast horizon
                                                                                                                                                                wsize = wsize, # the size of a rolling window
                                                                                                                                                                ylag = 1, # the number of lagged outcome variables
                                                                                                                                                                use_model = use_model, 
                                                                                                                                                                objective = objective, # a LGBM objective function
                                                                                                                                                                objective_name = f'{use_model}_{use_data}_{loss_fn_name}_{scoring_fn}_tau_{tau}', # the name of the objective function
                                                                                                                                                                scoring_fn = scoring_fn,  
                                                                                                                                                                use_custom_loss = use_custom_loss, # whether or not to use a custom loss function for LGBM
                                                                                                                                                                init_wealth = init_wealth, # an initial wealth
                                                                                                                                                                fixed_trans_cost = fixed_trans_cost, # the dollar amount of fixed transaction cost 
                                                                                                                                                                variable_trans_cost = variable_trans_cost, # an amount of transaction cost 
                                                                                                                                                                                                                                    #  as the percentage of stock price
                                                                                                                                                                n_trials = n_trials, # the number of trials set for Optuna
                                                                                                                                                                n_jobs = -1, # the number of workers for multiprocessing
                                                                                                                                                            )


        forecasts_df.to_csv(os.path.join(out_dir, \
            f'forecast_wsize_{wsize}_n_trials_{n_trials}_init_wealth_{init_wealth}_fixed_trans_cost_{fixed_trans_cost}_variable_trans_cost_{variable_trans_cost}.csv'), index=False, header = True) 
        scores_df.to_csv(os.path.join(out_dir, \
            f'scores_{wsize}_n_trials_{n_trials}_init_wealth_{init_wealth}_fixed_trans_cost_{fixed_trans_cost}_variable_trans_cost_{variable_trans_cost}.csv'), index=False, header = True) 
        opt_params_df.to_csv(os.path.join(out_dir, \
            f'opt_params_{wsize}_n_trials_{n_trials}_init_wealth_{init_wealth}_fixed_trans_cost_{fixed_trans_cost}_variable_trans_cost_{variable_trans_cost}.csv'), index=False, header = True) 
        importances_df.to_csv(os.path.join(out_dir, \
            f'importances_{wsize}_n_trials_{n_trials}_init_wealth_{init_wealth}_fixed_trans_cost_{fixed_trans_cost}_variable_trans_cost_{variable_trans_cost}.csv'), index=False, header = True) 
        SHAP_df.to_csv(os.path.join(out_dir, \
            f'SHAP_{wsize}_n_trials_{n_trials}_init_wealth_{init_wealth}_fixed_trans_cost_{fixed_trans_cost}_variable_trans_cost_{variable_trans_cost}.csv'), index=False, header = True) 

    end = time.time()
    print( 'Completed in: %s sec'%(end - start) )