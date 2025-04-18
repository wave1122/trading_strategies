# =============================Test the null hypothesis that the best of several performance measures is no superiority over a benchmark measure  =========================== #
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, List
import time
import os

# import the Reality Check (RC) module
from compare_models import do_reality_check_SB

if __name__ == "__main__":
    startTime = time.time()
    
    # use_models = ['RF_SPY_all_vars', 'RF_SPY_all_vars_plus_patterns']
    use_models = ['LGBM_SPY_all_vars', 'LGBM_SPY_all_vars_plus_patterns']
    
    # loss_fns = ['CE']
    loss_fns = ['CE', 'Brier', 'Boost', 'As1', 'As2']
    
    score_fns = ['Accuracy', 'AUC',  'Gain_to_pain_ratio_fixed_trans_cost', 'Calmar_ratio_fixed_trans_cost', \
                                                                                        'Sharpe_ratio_fixed_trans_cost',  'Sortino_ratio_fixed_trans_cost',  'CECPP_fixed_trans_cost']
    perf_metric = 'annualized_excess_return'
    holding_per = 100
    init_wealth = 1000
    fixed_trans_cost = 0.05
    
    # read all performance metrics from various methods, loss functions, scoring functions to a dataframe
    folder_path = Path(f'../Results/')
    perf_metrics =[]
    columns = []
    for use_model in use_models:
        for loss_fn in loss_fns:
            for score_fn in score_fns:
                file_path = Path.joinpath(folder_path, use_model, f'loss_fn={loss_fn}', f'score_fn={score_fn}','performance', \
                                                                                                                f'performance_hper_{holding_per}_init_wealth_{init_wealth}_fixed_trans_cost_{fixed_trans_cost}.csv')
                perf_df = pd.read_csv(file_path, encoding='utf-8', sep = ',', low_memory=False, header = 0, skiprows = 0, skipinitialspace=True, parse_dates = ['start_date', 'end_date'])
                columns.append(f'{use_model}_{loss_fn}_{score_fn}')
                perf_metrics.append(perf_df[perf_metric].tolist())
    perf_metrics_df = pd.DataFrame(np.array(perf_metrics).T, columns = columns)
    perf_metrics_df.insert(loc = 0, column = 'end_date', value = perf_df['end_date'])
    display( perf_metrics_df.head() )
    print(perf_metrics_df.shape)

    # create an output folder
    algo = re.search(r'.+(?=\_all)',  use_models[0], flags=re.IGNORECASE | re.VERBOSE).group()
    out_dir = Path.joinpath(folder_path, 'bootstrap', algo)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save data
    perf_metrics_df.to_csv(out_dir.joinpath('all_perf_metrics.csv'), index=False, header = True)

    # do the RC
    M = 999
    block = 50
    
    benchmarks, p_values_all = [], []
    for use_model in use_models:
        for loss_fn in loss_fns:
            for score_fn in score_fns:
                ## set a benchmark model
                benchmark = f'{use_model}_{loss_fn}_{score_fn}'
                benchmarks.append(benchmark)
                print('benchmark performance measure = ', benchmark)
                
                perf_A = perf_metrics_df.drop(columns = ['end_date', benchmark], axis = 1).values
                perf_B = perf_metrics_df[benchmark].values.reshape(-1, 1)
                        
                ## calculate the p-values
                p_values = do_reality_check_SB(  perf_A, # a N by K array of N values on each of K performance measures
                                                                        perf_B, # a N by 1 array of  N values on a benchmark performance measure
                                                                        M = M, # the number of sets of random observation indices
                                                                        block = block, # the mean block length
                                                                    )
                p_values_all.append(p_values)
        
    p_values_df = pd.DataFrame(np.array(p_values_all).T, columns = benchmarks)
    p_values_df.to_csv(out_dir.joinpath('all_p_values.csv'), index=False, header = True)


    print( 'The script took {} second !'.format(time.time() - startTime) )


            
 

# # List all filepaths in a folder

# file_paths = sorted([str(path) for path in folder_path.glob('*.csv')])
# print(file_paths)

# # Create a folder for output data
# out_dir = folder_path.joinpath('descriptive_stats')

# if not os.path.exists( out_dir ):
#     # Create the directory if it does not exist .
#     os.makedirs( out_dir )

# perf_df = pd.read_csv(file_paths[0], encoding='utf-8', sep = ',', low_memory=False, header = 0, skiprows = 0, skipinitialspace=True, parse_dates = ['start_date', 'end_date'])
# perf_df.drop(columns = ['start_date', 'end_date', 'ratio_profit_over_total_loss', 'annualized_return',	'annualized_return_bh'], inplace = True)

# get_hper = re.search(r'(?<=hper_)\d+(?=_)', file_paths[0], flags=re.IGNORECASE | re.VERBOSE).group()
# get_strategy = re.search(r'(?<=1000_)\w+(?=_\d+)', file_paths[0], flags=re.IGNORECASE | re.VERBOSE).group()
# get_trans_cost = re.search(r'(?<=\_)\d+\.?\d*(?=\.csv)', file_paths[0], flags=re.IGNORECASE | re.VERBOSE).group()





