import pandas as pd
import numpy as np
import os
import time
from varname import nameof

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# suppress warnings
warnings.filterwarnings('ignore')

import my_classification_algorithms_optuna_v3 as optuna_cl
import forecast_directional_movement_v1 as forecast
import custom_losses as closs

if __name__ == "__main__":
    data_df = pd.read_csv('../Data/RW_all_vars_plus_patterns.csv', encoding='utf-8', sep = ',', low_memory=False, header = 0, skiprows = 0, skipinitialspace=True, parse_dates = ['date'])
    data_df.set_index('date', inplace = True)
    # display( data_df.head() )

    use_data = 'RW_all_vars_plus_patterns'

    use_model = 'LGBM'

    use_custom_loss = True # whether or not to use a custom loss function for LGBM

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
        n_trials = 100 # set the number of trials for Optuna
    else:
        print(f'Model {use_model} does not exist!')
        sys.exit()

    # List scoring functions used to rank models during the cross validation. 
    # This argument can takes: ['Accuracy', 'Average_Precision', 'Precision', 'F1_score', 'AUC', 'Cross_entropy', 'As1_score', 'As2_score', 'Boost_score', 'Brier_score', 
                                                # 'Gain_to_pain_ratio_fixed_trans_cost', 'Gain_to_pain_ratio_variable_trans_cost', 'Calmar_ratio_fixed_trans_cost', 'Calmar_ratio_variable_trans_cost', 
                                                # 'Sharpe_ratio_fixed_trans_cost', 'Sharpe_ratio_variable_trans_cost', 'Sortino_ratio_fixed_trans_cost', 'Sortino_ratio_variable_trans_cost', 
                                                # 'CECPP_fixed_trans_cost', 'CECPP_variable_trans_cost']
    # scoring_fns = ['Accuracy', 'AUC', 'Gain_to_pain_ratio_fixed_trans_cost', 'Gain_to_pain_ratio_variable_trans_cost', 'Sharpe_ratio_fixed_trans_cost', \
                             # 'Sharpe_ratio_variable_trans_cost', 'Sortino_ratio_fixed_trans_cost', 'Sortino_ratio_variable_trans_cost', 'CECPP_fixed_trans_cost', 'CECPP_variable_trans_cost']   

    # scoring_fns = ['Calmar_ratio_fixed_trans_cost', 'Calmar_ratio_variable_trans_cost']   
    
    scoring_fns = ['Gain_to_pain_ratio_fixed_trans_cost']

    wsize = 1000 # set the rolling window size
    init_wealth = 1000 # an initial wealth
    fixed_trans_cost = 10 # the dollar amount of fixed transaction cost 
    variable_trans_cost = 0.005 # an amount of transaction cost as the percentage of stock price

    # Start timing the main loop
    start = time.time()

    for scoring_fn in scoring_fns:
        out_dir = f'./Results/{use_model}_{use_data}/loss_fn={loss_fn_name}/score_fn={scoring_fn}/'
        if not os.path.exists( out_dir ):
        # Create the directory if it does not exist .
            os.makedirs( out_dir )

        forecasts_df, scores_df, opt_params_df, importances_df, SHAP_df = forecast.rolling_forecast(data_df,
                                                                                                                                                                tau = 1, # a forecast horizon
                                                                                                                                                                wsize = wsize, # the size of a rolling window
                                                                                                                                                                ylag = 1, # the number of lagged outcome variables
                                                                                                                                                                use_model = use_model, 
                                                                                                                                                                objective = objective, # a LGBM objective function
                                                                                                                                                                objective_name = f'{use_model}_{use_data}_{loss_fn_name}_{scoring_fn}', # the name of the objective function
                                                                                                                                                                scoring_fn = scoring_fn,  
                                                                                                                                                                use_custom_loss = use_custom_loss, # whether or not to use a custom loss function for LGBM
                                                                                                                                                                init_wealth = init_wealth, # an initial wealth
                                                                                                                                                                fixed_trans_cost = fixed_trans_cost, # the dollar amount of fixed transaction cost 
                                                                                                                                                                variable_trans_cost = variable_trans_cost, # an amount of transaction cost 
                                                                                                                                                                                                                                    #  as the percentage of stock price
                                                                                                                                                                n_trials = n_trials, # the number of trials set for Optuna
                                                                                                                                                                n_jobs = 50, # the number of workers for multiprocessing
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