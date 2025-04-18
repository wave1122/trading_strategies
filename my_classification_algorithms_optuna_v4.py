# =========================================== ML Algorithms to Train/Validate/Forecast ==================================================== #
# ============================================================================================================================== #
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
        
import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels import robust
from statsmodels.tsa.api import VAR
from statistics import median
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# import Optuna
import optuna
from optuna.integration import LightGBMPruningCallback
from optuna.trial import TrialState
from optuna.samplers import TPESampler, RandomSampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
optuna.logging.set_verbosity(optuna.logging.WARNING)

# import Sklearn modules to train machine-learning models
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.inspection import permutation_importance

# import SciPy modules
from scipy.misc import derivative
from scipy.special import softmax

# import a tupling module
from typing import Tuple

# import SHAP module for model interpretation
import shap

# import data pre-processing modules
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# import Random Forest module
from sklearn.ensemble import RandomForestClassifier

# import LightGBM
import lightgbm as lgbm
from lightgbm import LGBMClassifier, plot_importance

# import CatBoost
from catboost import CatBoostClassifier

# import XGBoost module
import xgboost as xgb
from xgboost import XGBClassifier, XGBRFClassifier
from skopt import gp_minimize
from skopt.space import Real, Integer
from functools import partial

import logging
from suppress_stdout_stderr import suppress_stdout_stderr

# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import InputLayer, Dense, Activation, LSTM, Dropout, SimpleRNN

# from tensorflow.python.layers.normalization import Normalization
# from tensorflow.python.keras.layers.normalization import BatchNormalization
# from keras.layers.preprocessing.normalization import Normalization
# from keras.layers.normalization.batch_normalization import BatchNormalization

# from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.backend import clear_session
# from tensorflow.python.keras import regularizers
# from tensorflow.python.keras.layers import Flatten
# from tensorflow.python.keras.layers import Conv1D
# from tensorflow.python.keras.layers import MaxPooling1D
# from tensorflow.python.keras.metrics import AUC
# from tensorflow.keras.utils import to_categorical

import os
import sys
import gc
import math
from varname import nameof

import datetime
import time
import psutil
import multiprocessing as multi
from itertools import product, repeat
from functools import partial

# import custom loss functions
import custom_losses_v1 as closs

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
# tf.random.set_seed(SEED)

# Define the sigmoid function
def sigmoid(x): return 1. / (1. +  np.exp(-x))

# Compute average SHAP values for a ML model
def ave_SHAP_values(model, X: pd.DataFrame, use_method = 'tree_based'):
    """
        INPUT
            model: a trained model
            X: a pandas dataframe of features
        OUTPUT:
            a sorted dataframe of average SHAP values
    """
    if use_method == 'tree_based':
        shap_explainer = shap.TreeExplainer(model = model)
        shap_values_ = shap_explainer.shap_values(X)
        if isinstance(shap_values_, list):
            shap_values_ = shap_values_[1]

        shap_imps = pd.DataFrame({'feature': X.columns, 'average_SHAP_value': np.abs(shap_values_).mean(axis = 0)}) #.sort_values(by = 'average_SHAP_value', ascending = False)
        shap_imps['average_SHAP_value_norm'] = softmax(shap_imps.average_SHAP_value.values) # normalize feature importance scores with the softmax function
    elif use_method == 'deep_learning':
        shap_explainer = shap.KernelExplainer(model = model.predict_proba, data = shap.kmeans(X.values, 10))
        n_features = X.shape[1]
        shap_values_ = np.empty( shape = (0, n_features) )
        for i in np.random.choice(X.shape[0], 50, replace = False): # take a random subset of observations
            shap_values_i = shap_explainer.shap_values(X.values[i, :], nsamples = 100)
            shap_values_ = np.append(shap_values_, np.mean(np.abs(shap_values_i), axis = 0).reshape(1, n_features), axis = 0)
        
        shap_imps = pd.DataFrame({'feature': X.columns, 'average_SHAP_value': np.mean(shap_values_, axis = 0)}) # .sort_values(by = 'average_SHAP_value', ascending = False)
    else:
        print(f'Method {use_method} does not exist!')
        sys.exit()

    return shap_imps

# Create the box plot of permutation importance scores for a ML model
def box_plot_per_feats(model, X: pd.DataFrame, y: pd.Series, scoring = 'roc_auc') -> Tuple[pd.DataFrame, np.array]:
    """
        Compute and draw a box-plot of permutation importance scores
        INPUT
            model: a trained model
            X: a pandas dataframe of features
        OUTPUT
            a sorted dataframe of feature importance scores and their box plot
    """
    permutation_imp = permutation_importance(model, X, y, n_jobs = 1, scoring = scoring, n_repeats = 20, random_state = SEED)
    sorted_importances_idx = permutation_imp.importances_mean[permutation_imp.importances_mean > 0].argsort()
    importance_df = pd.DataFrame(permutation_imp.importances[sorted_importances_idx].T, columns=X.columns[sorted_importances_idx])

    fig, ax = plt.subplots( figsize=(13, 8) )
    importance_df.plot.box(vert=False, whis=10, ax = ax)
    ax.grid(ls=':')
    ax.legend(fontsize=15)
    ax.set_title("Permutation Importances")
    ax.axvline(x=0, linestyle="--", linewidth = 2, color ='red')
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    return importance_df, ax

# ======================================================== Code for XGBoost forecast ================================================================ #
# Create a XGBoost model
def create_XGB_model( booster = 'gbtree', 
                                        objective = 'binary:logistic', # objective function
                                        eval_metric = 'logloss', # evaluation metric
                                        n_estimators = 100, # number of gradient boosted trees
                                        learning_rate = 1., # learning rate of the gradient descent optimization algorithm
                                        max_depth = 6, # maximum depth of a tree
                                        min_child_weight = 1, # minimum sum of instance weight (hessian) needed in a child
                                        gamma = 0., # minimum loss reduction required to make a further partition on a leaf node of the tree
                                        colsample_bytree = 1., # subsample ratio of columns when constructing each tree
                                        subsample = 1.): # subsample ratio of the training instances
    model = XGBClassifier(booster=booster, objective = objective, eval_metric = eval_metric, n_estimators=n_estimators, learning_rate=learning_rate, \
                                            max_depth=max_depth, min_child_weight=min_child_weight, gamma=gamma, colsample_bytree=colsample_bytree, \
                                            subsample=subsample, use_label_encoder=False, seed=SEED, n_jobs = 30, verbosity = 0)
    return model

# Forecast with XGBoost
def XGBf(   R: pd.DataFrame, # a dataframe of class labels
                    X: np.array, # a numpy array of features with columns as features
                    data_df, # a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags
                    feat_names: list, # list of feature names
                    tau: int, # a forecast horizon
                    scoring_fn = 'AUC',  # a scoring function used to cross validate a LightGBM model 
                                                        # This argument can takes: ['Accuracy', 'Average_Precision', 'Precision', 'F1_score', 'AUC', 'Cross_entropy', 'As1_score', 'As2_score', 'Boost_score', 'Brier_score', 
                                                        # 'Gain_to_pain_ratio_fixed_trans_cost', 'Gain_to_pain_ratio_variable_trans_cost', 'Calmar_ratio_fixed_trans_cost', 'Calmar_ratio_variable_trans_cost', 
                                                        # 'Sharpe_ratio_fixed_trans_cost', 'Sharpe_ratio_variable_trans_cost', 'Sortino_ratio_fixed_trans_cost', 'Sortino_ratio_variable_trans_cost', 
                                                        # 'CECPP_fixed_trans_cost', 'CECPP_variable_trans_cost']
                    init_wealth = 1000, # an initial wealth
                    fixed_trans_cost = 10, # the dollar amount of fixed transaction cost 
                    variable_trans_cost = 0.005, # an amount of transaction cost as the percentage of stock price
                    n_trials = 100, # the number of trials set for Optuna
                    n_jobs = 1, # the number of workers for multiprocessing
            ):
    ''' Forecast with XGBoost
    '''
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    assert(tau > 0), "the forecast horizon must be greater than zero!"
    
    T = X.shape[0]
    dim = X.shape[1]

    R1 = R.shift(periods = -tau) # shift the labels forward

    # # Studentize data
    # scaler = StandardScaler()
    # X1 = scaler.fit_transform(X1)

    # split the numpy array into train and test data
    X1_train, X1_test = X[0:(T-tau), :], X[(T-tau):, :]
    R1_train, R1_test = R1.iloc[0: (T-tau)], R1.iloc[(T-tau):]


    # Rescale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X1_train)
    X1_train = scaler.transform(X1_train)
    X1_test = scaler.transform(X1_test)

    # Build a XGB model
    model = create_XGB_model()

    # Define the grid search parameters (cf. https://xgboost.readthedocs.io/en/stable/parameter.html)
    objective = optuna.distributions.CategoricalDistribution(['binary:logistic'])
    eval_metric = optuna.distributions.CategoricalDistribution(['logloss'])  # an evaluation metric: 'auc' or 'logloss'
    booster = optuna.distributions.CategoricalDistribution(['gbtree'])
    n_estimators = optuna.distributions.IntDistribution(10, 200, step = 10) # number of gradient boosted trees
    learning_rate = optuna.distributions.UniformDistribution(0., 100.) # learning rate of the gradient descent optimization algorithm
    max_depth = optuna.distributions.IntDistribution(1, 6) # maximum depth of a tree
    min_child_weight = optuna.distributions.IntDistribution(1, 20)  # minimum sum of instance weight (hessian) needed in a child
    gamma = optuna.distributions.UniformDistribution(0., 20.) # minimum loss reduction required to make a further partition on a leaf node of the tree
    colsample_bytree = optuna.distributions.UniformDistribution(0., 1.) # subsample ratio of columns used to choose the best splitting point in each tree
    subsample =optuna.distributions.UniformDistribution(0., 1.) # subsample ratio of training examples

    param_grid = dict(objective = objective, eval_metric = eval_metric, booster = booster, n_estimators = n_estimators, learning_rate = learning_rate, \
                                max_depth = max_depth, min_child_weight = min_child_weight, gamma = gamma, colsample_bytree = colsample_bytree, \
                                subsample = subsample)

    # List score functions used to train models. Note the convention that higher score values are better than lower score values
    gain_to_pain_ratio_score_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.gain_to_pain_ratio_score_sklearn(  y_true, 
                                                                                                                                                                                                    y_prob,
                                                                                                                                                                                                    data_df = data_df,
                                                                                                                                                                                                    init_wealth = init_wealth, 
                                                                                                                                                                                                    use_fixed_trans_cost = True, 
                                                                                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                                    variable_trans_cost = variable_trans_cost)
    gain_to_pain_ratio_score_fixed_trans_cost_sklearn = make_scorer(gain_to_pain_ratio_score_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
    gain_to_pain_ratio_score_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.gain_to_pain_ratio_score_sklearn( y_true, 
                                                                                                                                                                                                        y_prob,
                                                                                                                                                                                                        data_df,
                                                                                                                                                                                                        init_wealth = init_wealth, 
                                                                                                                                                                                                        use_fixed_trans_cost = False, 
                                                                                                                                                                                                        fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                                        variable_trans_cost = variable_trans_cost)
    gain_to_pain_ratio_score_variable_trans_cost_sklearn = make_scorer(gain_to_pain_ratio_score_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

    calmar_ratio_score_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.calmar_ratio_score_sklearn(  y_true, 
                                                                                                                                                                                y_prob,
                                                                                                                                                                                data_df = data_df,
                                                                                                                                                                                init_wealth = init_wealth, 
                                                                                                                                                                                use_fixed_trans_cost = True, 
                                                                                                                                                                                fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                variable_trans_cost = variable_trans_cost)
    calmar_ratio_score_fixed_trans_cost_sklearn = make_scorer(calmar_ratio_score_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
    calmar_ratio_score_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.calmar_ratio_score_sklearn( y_true, 
                                                                                                                                                                                    y_prob,
                                                                                                                                                                                    data_df = data_df,
                                                                                                                                                                                    init_wealth = init_wealth, 
                                                                                                                                                                                    use_fixed_trans_cost = False, 
                                                                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                    variable_trans_cost = variable_trans_cost)
    calmar_ratio_score_variable_trans_cost_sklearn = make_scorer(calmar_ratio_score_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

    sharpe_ratio_score_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.sharpe_ratio_score_sklearn(  y_true, 
                                                                                                                                                                                y_prob,
                                                                                                                                                                                data_df = data_df,
                                                                                                                                                                                init_wealth = init_wealth, 
                                                                                                                                                                                use_fixed_trans_cost = True, 
                                                                                                                                                                                fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                variable_trans_cost = variable_trans_cost)
    sharpe_ratio_score_fixed_trans_cost_sklearn = make_scorer(sharpe_ratio_score_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
    sharpe_ratio_score_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.sharpe_ratio_score_sklearn( y_true, 
                                                                                                                                                                                    y_prob,
                                                                                                                                                                                    data_df = data_df,
                                                                                                                                                                                    init_wealth = init_wealth, 
                                                                                                                                                                                    use_fixed_trans_cost = False, 
                                                                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                    variable_trans_cost = variable_trans_cost)
    sharpe_ratio_score_variable_trans_cost_sklearn = make_scorer(sharpe_ratio_score_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

    sortino_ratio_score_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.sortino_ratio_score_sklearn(  y_true, 
                                                                                                                                                                                y_prob,
                                                                                                                                                                                data_df = data_df,
                                                                                                                                                                                init_wealth = init_wealth, 
                                                                                                                                                                                use_fixed_trans_cost = True, 
                                                                                                                                                                                fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                variable_trans_cost = variable_trans_cost)
    sortino_ratio_score_fixed_trans_cost_sklearn = make_scorer(sortino_ratio_score_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
    sortino_ratio_score_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.sortino_ratio_score_sklearn( y_true, 
                                                                                                                                                                                    y_prob,
                                                                                                                                                                                    data_df = data_df,
                                                                                                                                                                                    init_wealth = init_wealth, 
                                                                                                                                                                                    use_fixed_trans_cost = False, 
                                                                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                    variable_trans_cost = variable_trans_cost)
    sortino_ratio_score_variable_trans_cost_sklearn = make_scorer(sortino_ratio_score_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

    cecpp_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.cecpp_sklearn( y_true, 
                                                                                                                                    y_prob,
                                                                                                                                    data_df = data_df,
                                                                                                                                    init_wealth = init_wealth, 
                                                                                                                                    use_fixed_trans_cost = True, 
                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                    variable_trans_cost = variable_trans_cost)
    cecpp_fixed_trans_cost_sklearn = make_scorer(cecpp_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
    cecpp_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.cecpp_sklearn(y_true, 
                                                                                                                                        y_prob,
                                                                                                                                        data_df = data_df,
                                                                                                                                        init_wealth = init_wealth, 
                                                                                                                                        use_fixed_trans_cost = False, 
                                                                                                                                        fixed_trans_cost = fixed_trans_cost,
                                                                                                                                        variable_trans_cost = variable_trans_cost)
    cecpp_variable_trans_cost_sklearn = make_scorer(cecpp_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

    scoring = {'Accuracy': 'accuracy', 'Average_Precision': 'average_precision', 'Precision': 'precision',  'F1_score': 'f1', 'AUC': 'roc_auc', 'Cross_entropy': 'neg_log_loss', \
                        'As1_score': closs.As1_score_sklearn, 'As2_score': closs.As2_score_sklearn, 'Boost_score': closs.Boost_score_sklearn, 'Brier_score': closs.Brier_score_sklearn,  \
                        'Gain_to_pain_ratio_fixed_trans_cost': gain_to_pain_ratio_score_fixed_trans_cost_sklearn, \
                        'Gain_to_pain_ratio_variable_trans_cost': gain_to_pain_ratio_score_variable_trans_cost_sklearn, \
                        'Calmar_ratio_fixed_trans_cost': calmar_ratio_score_fixed_trans_cost_sklearn, 'Calmar_ratio_variable_trans_cost': calmar_ratio_score_variable_trans_cost_sklearn, \
                        'Sharpe_ratio_fixed_trans_cost': sharpe_ratio_score_fixed_trans_cost_sklearn, 'Sharpe_ratio_variable_trans_cost': sharpe_ratio_score_variable_trans_cost_sklearn, \
                        'Sortino_ratio_fixed_trans_cost': sortino_ratio_score_fixed_trans_cost_sklearn, 'Sortino_ratio_variable_trans_cost': sortino_ratio_score_variable_trans_cost_sklearn, \
                        'CECPP_fixed_trans_cost': cecpp_fixed_trans_cost_sklearn, 'CECPP_variable_trans_cost': cecpp_variable_trans_cost_sklearn}  

    # use the time-series cross-validation
    # tscv = TimeSeriesSplit(n_splits = 2, test_size = 10)

    # use the stratified K-fold cross-validation
    skfcv = StratifiedKFold(n_splits = 5, shuffle=True, random_state = SEED)

    # Create a new Optuna study
    sampler = TPESampler(seed=SEED) # perform Bayesian optimization hyperparameter tuning
    # sampler = RandomSampler(seed=SEED) # perform Random Search
    study = optuna.create_study( direction = "maximize", # minimize a loss function
                                                    study_name = "XGBoost Classifier",
                                                    pruner=optuna.pruners.HyperbandPruner(),
                                                    sampler = sampler)

	# Perform grid search
    model_cv = optuna.integration.OptunaSearchCV(model, param_distributions = param_grid, cv = skfcv, study = study, scoring = scoring[scoring_fn], refit = True, n_jobs = n_jobs, n_trials = n_trials)

    # Cross-validate a model by using the grid search
    model_cv.fit(X1_train, R1_train, verbose = 0)
    
    # # Plot the optimization history
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.update_layout(autosize=False, width=1100, height=600)
    # fig.update_xaxes(automargin=True)
    # display(fig)
    
    # # Generate the coordinate plot of the hyperparameter tuning process
    # fig = plot_parallel_coordinate(study)
    # fig.update_layout(autosize=False, width=1100, height=600)
    # fig.update_xaxes(automargin=True)
    # display(fig)

    # Retrieve the best model
    best_model = model_cv.best_estimator_

    # Forecast the test data
    forecasts = model_cv.predict_proba(X1_test)[:, 1]

    if tau > 1:
        forecast_tau = forecasts[len(forecasts)-1]
    else:
        forecast_tau = forecasts

    # Get the optimal hyperparameters
    opt_params = model_cv.best_params_
    # print(f'Optimal hyperparameters:\n {opt_params}')

    # Calculate the feature importance scores based on the best model fitted to the training sample
    features_importances_df = pd.DataFrame({'feature': feat_names, 'importance_score': best_model.feature_importances_}).round(decimals=2)\
                                                                                                                                                    # .sort_values(by = 'importance_score', ascending=False)
    # display(features_importances_df)

    # Calculate average SHAP values based on the best model fitted to the training sample
    X1_train_df = pd.DataFrame(X1_train, columns = feat_names)
    ave_SHAP_vlues_df = ave_SHAP_values(best_model, X1_train_df)
    # display(ave_SHAP_vlues_df)

    # Calculate the classification scores for the validation data
    scores = cross_validate(best_model, X1_train, R1_train, cv = skfcv, scoring = scoring, n_jobs = 30)
    scores_median = {key: np.median(v[~np.isnan(v)]) for key, v in scores.items()}

    del model, model_cv, best_model # delete all models
    gc.collect()

    # output results
    return float(forecast_tau), scores_median, opt_params, features_importances_df, ave_SHAP_vlues_df

# ======================================================= End of XGBoost Code ====================================================================== #

# ======================================================= Code for Random Forest forecast ============================================================== #
# Create a Random Forest model
def create_RF_model(n_estimators = 100, # number of decision trees used to build the random forest
                                    max_features = 'sqrt', # number of randomly sampled features that are used by RF to choose the best splitting point in each tree
                                    max_depth = None, # maximum depth of each decision tree (set it to a lower value to prevent overfitting)
                                    min_samples_split = 2, # minimum number of samples required for a tree to be able to further split an internal node (set it to a higher value to prevent overfitting)
                                    min_samples_leaf = 1, # minimum number of samples required in a leaf node (set it to a higher value to prevent overfitting)
                                    ):
    model =  RandomForestClassifier( n_estimators = n_estimators, max_features = max_features, max_depth = max_depth, min_samples_split = min_samples_split, 
                                                            min_samples_leaf = min_samples_leaf, random_state = SEED, n_jobs = 20, verbose = 0)
    return model

# Forecast with Random Forest
def RFf(  R: pd.DataFrame, # a dataframe of class labels
                X: np.array, # a numpy array of features with columns as features
                data_df: pd.DataFrame, # a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags
                feat_names: list, # list of feature names
                tau: int, # a forecast horizon
                scoring_fn = 'AUC',  # a scoring function used to cross validate a LightGBM model 
                                                    # This argument can takes: ['Accuracy', 'Average_Precision', 'Precision', 'F1_score', 'AUC', 'Cross_entropy', 'As1_score', 'As2_score', 'Boost_score', 'Brier_score', 
                                                    # 'Gain_to_pain_ratio_fixed_trans_cost', 'Gain_to_pain_ratio_variable_trans_cost', 'Calmar_ratio_fixed_trans_cost', 'Calmar_ratio_variable_trans_cost', 
                                                    # 'Sharpe_ratio_fixed_trans_cost', 'Sharpe_ratio_variable_trans_cost', 'Sortino_ratio_fixed_trans_cost', 'Sortino_ratio_variable_trans_cost', 
                                                    # 'CECPP_fixed_trans_cost', 'CECPP_variable_trans_cost']
                init_wealth = 1000, # an initial wealth
                fixed_trans_cost = 10, # the dollar amount of fixed transaction cost 
                variable_trans_cost = 0.005, # an amount of transaction cost as the percentage of stock price
                n_trials = 100, # the number of trials set for Optuna
                n_jobs = 1, # the number of workers for multiprocessing
            ):
    ''' Forecast with Random Forest
    '''
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    assert(tau > 0), "the forecast horizon must be greater than zero!"
    
    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = R.shift(periods = -tau) # shift the labels forward

    # # Studentize data
    # scaler = StandardScaler()
    # X1 = scaler.fit_transform(X1)

    # split the numpy array into train and test data
    X1_train, X1_test = X[0:(T-tau), :], X[(T-tau):, :]
    R1_train, R1_test = R1.iloc[0: (T-tau)], R1.iloc[(T-tau):]

     # Rescale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X1_train)
    X1_train = scaler.transform(X1_train)
    X1_test = scaler.transform(X1_test)

    # Build a Random Forest model
    model = create_RF_model()

    # Define the grid search parameters (cf. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    n_estimators = optuna.distributions.IntDistribution(10, 200, step = 10) # number of gradient boosted trees
    max_features = optuna.distributions.IntDistribution(1, dim) # number of randomly sampled features that are used by RF to choose the best splitting point in each tree
    max_depth = optuna.distributions.IntDistribution(1, 6) # maximum depth of a tree  (set it to a lower value to prevent overfitting)
    min_samples_split = optuna.distributions.IntDistribution(2, 102, step = 10) # minimum number of samples required for a tree to be able to further split an internal node  
                                                                                                                                           # (set it to a higher value to prevent overfitting)
    min_samples_leaf = optuna.distributions.IntDistribution(1, 101, step = 10) # minimum number of samples required in a leaf node (set it to a higher value to prevent overfitting)

    param_grid = dict(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)

    # List score functions used to train models. Note the convention that higher score values are better than lower score values
    gain_to_pain_ratio_score_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.gain_to_pain_ratio_score_sklearn(  y_true, 
                                                                                                                                                                                                    y_prob,
                                                                                                                                                                                                    data_df = data_df,
                                                                                                                                                                                                    init_wealth = init_wealth, 
                                                                                                                                                                                                    use_fixed_trans_cost = True, 
                                                                                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                                    variable_trans_cost = variable_trans_cost)
    gain_to_pain_ratio_score_fixed_trans_cost_sklearn = make_scorer(gain_to_pain_ratio_score_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
    gain_to_pain_ratio_score_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.gain_to_pain_ratio_score_sklearn( y_true, 
                                                                                                                                                                                                        y_prob,
                                                                                                                                                                                                        data_df = data_df,
                                                                                                                                                                                                        init_wealth = init_wealth, 
                                                                                                                                                                                                        use_fixed_trans_cost = False, 
                                                                                                                                                                                                        fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                                        variable_trans_cost = variable_trans_cost)
    gain_to_pain_ratio_score_variable_trans_cost_sklearn = make_scorer(gain_to_pain_ratio_score_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

    calmar_ratio_score_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.calmar_ratio_score_sklearn(  y_true, 
                                                                                                                                                                                y_prob,
                                                                                                                                                                                data_df = data_df,
                                                                                                                                                                                init_wealth = init_wealth, 
                                                                                                                                                                                use_fixed_trans_cost = True, 
                                                                                                                                                                                fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                variable_trans_cost = variable_trans_cost)
    calmar_ratio_score_fixed_trans_cost_sklearn = make_scorer(calmar_ratio_score_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
    calmar_ratio_score_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.calmar_ratio_score_sklearn( y_true, 
                                                                                                                                                                                    y_prob,
                                                                                                                                                                                    data_df = data_df,
                                                                                                                                                                                    init_wealth = init_wealth, 
                                                                                                                                                                                    use_fixed_trans_cost = False, 
                                                                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                    variable_trans_cost = variable_trans_cost)
    calmar_ratio_score_variable_trans_cost_sklearn = make_scorer(calmar_ratio_score_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

    sharpe_ratio_score_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.sharpe_ratio_score_sklearn(  y_true, 
                                                                                                                                                                                y_prob,
                                                                                                                                                                                data_df = data_df,
                                                                                                                                                                                init_wealth = init_wealth, 
                                                                                                                                                                                use_fixed_trans_cost = True, 
                                                                                                                                                                                fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                variable_trans_cost = variable_trans_cost)
    sharpe_ratio_score_fixed_trans_cost_sklearn = make_scorer(sharpe_ratio_score_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
    sharpe_ratio_score_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.sharpe_ratio_score_sklearn( y_true, 
                                                                                                                                                                                    y_prob,
                                                                                                                                                                                    data_df = data_df,
                                                                                                                                                                                    init_wealth = init_wealth, 
                                                                                                                                                                                    use_fixed_trans_cost = False, 
                                                                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                    variable_trans_cost = variable_trans_cost)
    sharpe_ratio_score_variable_trans_cost_sklearn = make_scorer(sharpe_ratio_score_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

    sortino_ratio_score_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.sortino_ratio_score_sklearn(  y_true, 
                                                                                                                                                                                y_prob,
                                                                                                                                                                                data_df = data_df,
                                                                                                                                                                                init_wealth = init_wealth, 
                                                                                                                                                                                use_fixed_trans_cost = True, 
                                                                                                                                                                                fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                variable_trans_cost = variable_trans_cost)
    sortino_ratio_score_fixed_trans_cost_sklearn = make_scorer(sortino_ratio_score_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
    sortino_ratio_score_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.sortino_ratio_score_sklearn( y_true, 
                                                                                                                                                                                    y_prob,
                                                                                                                                                                                    data_df = data_df,
                                                                                                                                                                                    init_wealth = init_wealth, 
                                                                                                                                                                                    use_fixed_trans_cost = False, 
                                                                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                    variable_trans_cost = variable_trans_cost)
    sortino_ratio_score_variable_trans_cost_sklearn = make_scorer(sortino_ratio_score_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

    cecpp_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.cecpp_sklearn( y_true, 
                                                                                                                                    y_prob,
                                                                                                                                    data_df = data_df,
                                                                                                                                    init_wealth = init_wealth, 
                                                                                                                                    use_fixed_trans_cost = True, 
                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                    variable_trans_cost = variable_trans_cost)
    cecpp_fixed_trans_cost_sklearn = make_scorer(cecpp_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
    cecpp_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.cecpp_sklearn(y_true, 
                                                                                                                                        y_prob,
                                                                                                                                        data_df = data_df,
                                                                                                                                        init_wealth = init_wealth, 
                                                                                                                                        use_fixed_trans_cost = False, 
                                                                                                                                        fixed_trans_cost = fixed_trans_cost,
                                                                                                                                        variable_trans_cost = variable_trans_cost)
    cecpp_variable_trans_cost_sklearn = make_scorer(cecpp_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

    scoring = {'Accuracy': 'accuracy', 'Average_Precision': 'average_precision', 'Precision': 'precision',  'F1_score': 'f1', 'AUC': 'roc_auc', 'Cross_entropy': 'neg_log_loss', \
                        'As1_score': closs.As1_score_sklearn, 'As2_score': closs.As2_score_sklearn, 'Boost_score': closs.Boost_score_sklearn, 'Brier_score': closs.Brier_score_sklearn,  \
                        'Gain_to_pain_ratio_fixed_trans_cost': gain_to_pain_ratio_score_fixed_trans_cost_sklearn, \
                        'Gain_to_pain_ratio_variable_trans_cost': gain_to_pain_ratio_score_variable_trans_cost_sklearn, \
                        'Calmar_ratio_fixed_trans_cost': calmar_ratio_score_fixed_trans_cost_sklearn, 'Calmar_ratio_variable_trans_cost': calmar_ratio_score_variable_trans_cost_sklearn, \
                        'Sharpe_ratio_fixed_trans_cost': sharpe_ratio_score_fixed_trans_cost_sklearn, 'Sharpe_ratio_variable_trans_cost': sharpe_ratio_score_variable_trans_cost_sklearn, \
                        'Sortino_ratio_fixed_trans_cost': sortino_ratio_score_fixed_trans_cost_sklearn, 'Sortino_ratio_variable_trans_cost': sortino_ratio_score_variable_trans_cost_sklearn, \
                        'CECPP_fixed_trans_cost': cecpp_fixed_trans_cost_sklearn, 'CECPP_variable_trans_cost': cecpp_variable_trans_cost_sklearn}  

    # use time-series cross-validation
    # tscv = TimeSeriesSplit(n_splits = 2, test_size = 10) 

    # use the stratified K-fold cross-validation
    skfcv = StratifiedKFold(n_splits = 5, shuffle=True, random_state = SEED)

    # Create a new Optuna study
    sampler = TPESampler(seed=SEED) # perform Bayesian optimization hyperparameter tuning
    # sampler = RandomSampler(seed=SEED) # perform Random Search
    study = optuna.create_study(direction = "maximize", # minimize a loss function
                                                    study_name = "RF Classifier",
                                                    pruner=optuna.pruners.HyperbandPruner(),
                                                    sampler = sampler)

	# Perform grid search
    model_cv = optuna.integration.OptunaSearchCV(model, param_distributions = param_grid, cv = skfcv, study = study, scoring = scoring[scoring_fn], refit = True, n_jobs = n_jobs, n_trials = n_trials)


    # Cross-validate a model by Optuna
    model_cv.fit(X1_train, R1_train)
    
    # # Plot the optimization history
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.update_layout(autosize=False, width=1100, height=600)
    # fig.update_xaxes(automargin=True)
    # display(fig)
    
    # # Generate the coordinate plot of the hyperparameter tuning process
    # fig = plot_parallel_coordinate(study)
    # fig.update_layout(autosize=False, width=1100, height=600)
    # fig.update_xaxes(automargin=True)
    # display(fig)

    # Forecast the test data
    forecasts = model_cv.predict_proba(X1_test)[:, 1]
    if tau > 1:
        forecast_tau = forecasts[len(forecasts)-1]
    else:
        forecast_tau = forecasts

    # Get the optimal hyperparameters
    opt_params = model_cv.best_params_
    # print(f'Optimal hyperparameters:\n {opt_params}')

    # Get the best cross-validated model
    best_model = model_cv.best_estimator_

    # Calculate the feature importance scores based on the best model fitted to the training sample
    features_importances_df = pd.DataFrame({'feature': feat_names, 'importance_score': best_model.feature_importances_}).round(decimals=2)\
                                                                                                                                                    # .sort_values(by = 'importance_score', ascending=False)
    # display(features_importances_df)

    # Calculate average SHAP values based on the best model fitted to the training sample
    X1_train_df = pd.DataFrame(X1_train, columns = feat_names)
    ave_SHAP_vlues_df = ave_SHAP_values(best_model, X1_train_df)
    # display(ave_SHAP_vlues_df)

    # Calculate the classification scores for the validation data
    scores = cross_validate(best_model, X1_train, R1_train, cv = skfcv, scoring = scoring, n_jobs = 30)
    scores_median = {key: np.median(v[~np.isnan(v)]) for key, v in scores.items()}

    del model, model_cv, best_model # delete all models
    gc.collect()

    # output results
    return float(forecast_tau), scores_median, opt_params, features_importances_df, ave_SHAP_vlues_df

# ====================================================== End of Random Forest Code =================================================================== #

# ======================================================= Code for CatBoost forecast ============================================================== #
# Create a CatBoost model
def create_CatBoost_model(loss_function = closs.BrierlossObjective(), eval_metric = closs.BrierMetric(), iterations = 10, learning_rate = 1., depth = 10, l2_leaf_reg = 0.1,  \
                                                                                                        boosting_type = 'Plain', bootstrap_type = 'No',  subsample = 0.7, random_strength = 1., rsm = 1.):
    model =  CatBoostClassifier( loss_function = loss_function, # set a loss function: 'Logloss', 'CrossEntropy'
                                                    eval_metric = eval_metric, # set an evaluation metric: 'AUC', 'Accuracy'. (See https://catboost.ai/en/docs/concepts/loss-functions-classification)
                                                    iterations = iterations, # set the maximum number of trees that can be built when solving ML problems
                                                    learning_rate = learning_rate, # set the learning rate of the gradient descent optimization algorithm
                                                    depth = depth, # set the level of depth for each tree
                                                    l2_leaf_reg = l2_leaf_reg, # set the L2 regularization parameter on the cost function
                                                    # boosting_type = boosting_type, # set a boosting schema: 'Ordered', 'Plain'
                                                    # bootstrap_type = bootstrap_type, # set a method for sampling the weights of objects: 'Bayesian', 'Bernoulli', 'MVS', 'Poisson', 'No'
                                                    # subsample = subsample, # set a sample rate for bagging
                                                    # random_strength = random_strength, # set the amount of randomness used for scoring splits when the tree structure is selected
                                                    # rsm = rsm, # set the percentage of features (must be in (0,1]) to use at each split selection when features are selected over again at random
                                                    # early_stopping_rounds = 100, #  stop training after the specified number of iterations since the iteration with the optimal metric value.
                                                    # use_best_model=True,
                                                    random_seed = SEED,
                                                    verbose = False,
                                                ) 
    return model

# Forecast with CatBoost
def CatBoostf(R: np.array, # a numpy array of class labels
                        X: np.array, # a numpy array of features with columns as features
                        data_df: pd.DataFrame, # a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags
                        feat_names: list, # list of feature names
                        tau: int, # a forecast horizon
                        loss_function = closs.BrierlossObjective(), # a custom loss function
                        eval_metric = closs.BrierMetric(), # a custom evaluation metric
                        scoring_fn = 'AUC',  # a scoring function used to rank models during the cross validation. 
                                                          # This argument can takes: 'As1_score',  'As2_score', 'Boost_score', 'Brier_score'
                        use_custom_loss = True, # whether or not to use a custom loss function
                        n_trials = 200, # the number of trials used by Optuna
                        n_jobs = 1, # the maximum number of concurrently running workers
                        ):
    ''' Forecast with CatBoost 
    '''
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    assert(tau > 0), "the forecast horizon must be greater than zero!"
    
    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = np.empty( shape = (0, 1) )
    X1 = np.empty( shape = (0, dim) )
    for t in np.arange(0, T):
        if t < T-tau:
            R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
        else:
            R1 = np.append(R1, np.array([0]).reshape(1, 1), axis = 0)
        X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

    # # Studentize data
    # scaler = StandardScaler()
    # X1 = scaler.fit_transform(X1)

    # split the numpy array into train and test data
    X1_train, X1_test = X1[0:(T-tau), :], X1[(T-tau):, :]
    R1_train, R1_test = R1[0:(T-tau)].ravel(), R1[(T-tau):].ravel()

     # Rescale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X1_train)
    X1_train = scaler.transform(X1_train)
    X1_test = scaler.transform(X1_test)


    # Build a CatBoost model
    model = create_CatBoost_model(loss_function = loss_function, eval_metric = eval_metric)

        # Define the grid search parameters
    if not use_custom_loss:
        loss_function = optuna.distributions.CategoricalDistribution(['CrossEntropy'])
        eval_metric = optuna.distributions.CategoricalDistribution(['CrossEntropy']) # set the evaluation metric for cross-validation: 'CrossEntropy', 'AUC', 'Logloss'
                                                                                                                                   # (https://catboost.ai/en/docs/references/custom-metric__supported-metrics)
    iterations = optuna.distributions.IntDistribution(1, 100) # the maximum number of trees
    depth = optuna.distributions.IntDistribution(1, 5) # the maximum depth of a tree
    learning_rate = optuna.distributions.UniformDistribution(0., 1.) # the learning rate used for stochastic gradient descent
    l2_leaf_reg =  optuna.distributions.UniformDistribution(0., 1.)  # the L2 regularization parameter on the cost function
    boosting_type = optuna.distributions.CategoricalDistribution(["Ordered", "Plain"]) # a boosting scheme:
                                                                                                                                     # Ordered — Usually provides better quality on small datasets, but it may be slower than the Plain scheme
                                                                                                                                     # Plain — The classic gradient boosting scheme
    bootstrap_type = optuna.distributions.CategoricalDistribution(['Bayesian', 'Bernoulli', 'MVS']) # the methods to sample the weights of objects: 
                                                                                                                                                                            # https://catboost.ai/en/docs/concepts/algorithm-main-stages_bootstrap-options

    random_strength =  optuna.distributions.UniformDistribution(0., 100) # the amount of randomness to use for scoring splits 
    rsm = optuna.distributions.UniformDistribution(0., 1.) # the percentage of features to use at each split selection

    # # List score functions used to train models. Note the convention that higher score values are better than lower score values
    if use_custom_loss:
        scoring = {  'Accuracy': closs.accuracy_score_cb, 
                            'Average_Precision': closs.average_precision_score_cb, 
                            'Precision': closs.precision_score_cb, 
                            'F1_score': closs.f1_score_cb, \
                            'AUC': closs.roc_auc_cb, 
                            'As1_score': closs.As1_score_cb, 
                            'As2_score': closs.As2_score_cb, 
                            'Boost_score': closs.Boost_score_cb, 
                            'Brier_score': closs.Brier_score_cb } 
    else:
            scoring = {  'Accuracy': 'accuracy', 
                                'Average Precision': 'average_precision', 
                                'Precision': 'precision', 
                                'AUC': 'roc_auc', 
                                'As1_score': closs.As1_score_sklearn, 
                                'As2_score': closs.As2_score_sklearn, \
                                'Boost_score': closs.Boost_score_sklearn, 
                                'Brier_score': closs.Brier_score_sklearn}
																								

    if use_custom_loss: 
        param_grid = dict(  iterations = iterations, 
                                        learning_rate = learning_rate, 
                                        depth = depth, \
                                        l2_leaf_reg = l2_leaf_reg, 
                                        # boosting_type = boosting_type, 
                                        # bootstrap_type = bootstrap_type, 
                                        # random_strength = random_strength, 
                                        # rsm = rsm
                                    )
    else:
        param_grid = dict(  loss_function = loss_function, 
                                        eval_metric = eval_metric, 
                                        iterations = iterations, 
                                        learning_rate = learning_rate, 
                                        depth = depth, \
                                        l2_leaf_reg = l2_leaf_reg, 
                                        # boosting_type = boosting_type, 
                                        # bootstrap_type = bootstrap_type, 
                                        # random_strength = random_strength, 
                                        # rsm = rsm
                                    )

    # if param_grid["bootstrap_type"] == "Bayesian":
    #     param_grid["bagging_temperature"] = optuna.distributions.UniformDistribution(0., 10.)
    # if param_grid["bootstrap_type"] == "Bernoulli":
    #     param_grid["subsample"] = optuna.distributions.UniformDistribution(0.1, 1.)

	# use time-series cross-validation
    # tscv = TimeSeriesSplit(n_splits = 2, test_size = 10) 

    # use the stratified K-fold cross-validation
    skfcv = StratifiedKFold(n_splits = 5, shuffle=True, random_state = SEED)

    # Create a new Optuna study
    # sampler = TPESampler(seed=SEED) # perform Bayesian optimization hyperparameter tuning
    sampler = RandomSampler(seed=SEED) # perform Random Search
    study = optuna.create_study(direction = "maximize", # minimize a loss function
                                                    study_name = "CatBoost Classifier",
                                                    # pruner=optuna.pruners.HyperbandPruner(),
                                                    # pruner = optuna.pruners.SuccessiveHalvingPruner(reduction_factor=3, min_resource=5), # use Successive Halving as the pruner
                                                    pruner = optuna.pruners.PercentilePruner( percentile = 25.0, n_startup_trials = 5, n_warmup_steps = 0, interval_steps = 1, n_min_trials = 5),
                                                    sampler = sampler)

	# Perform grid search
    model_cv = optuna.integration.OptunaSearchCV(model, param_distributions = param_grid, cv = skfcv, study = study, scoring = scoring[scoring_fn], refit = True, n_jobs = n_jobs, n_trials = n_trials)



    # Cross-validate a model by using the grid search
    model_cv.fit(X1_train, R1_train, verbose = False)
    # R1_pred =  model_cv.predict_proba(X1_train)[:, 1]
    # # print('R1_pred = ', R1_pred)
    # R1_pred = (R1_pred > 0.5).astype('int')
    # R1_err = np.c_[R1_train, R1_pred]
    # np.savetxt('./R1_error.csv', R1_err, delimiter = ' , ')

    # Plot the hyperparameter tuning progress
    fig = plot_parallel_coordinate(study)
    fig.update_layout(autosize=False, width=1100, height=600)
    fig.update_xaxes(automargin=True)
    display(fig)

    # Forecast the test data
    forecasts = model_cv.predict_proba(X1_test)[:, 1]
    # forecasts = (forecasts > 0.5).astype('int')
    if tau > 1:
        forecast_tau = forecasts[len(forecasts)-1]
    else:
        forecast_tau = forecasts

    # Get the optimal hyperparameters
    opt_params = model_cv.best_params_
   # print(f'Optimal hyperparameters:\n {opt_params}')

    # Calculate the feature importance scores based on the best model fitted to the training sample
    best_model = model_cv.best_estimator_
    features_importances_df = pd.DataFrame({'feature': feat_names, 'importance_score': best_model.get_feature_importance()}).round(decimals=2)\
                                                                                                                                                    # .sort_values(by = 'importance_score', ascending=False)
    # display(features_importances_df)

    # Calculate average SHAP values based on the best model fitted to the training sample
    X1_train_df = pd.DataFrame(X1_train, columns = feat_names)
    ave_SHAP_vlues_df = ave_SHAP_values(best_model, X1_train_df)
    # display(ave_SHAP_vlues)

    
    # Calculate the classification scores for the validation data
    scores = cross_validate(best_model, X1_train, R1_train, cv = skfcv, scoring = scoring, n_jobs = 1, verbose = 0)
    scores_mean = {key: np.mean(v) for key, v in scores.items()}

    del model, model_cv, best_model  # delete all models
    gc.collect()

    # output results
    return float(forecast_tau), scores_mean, opt_params, features_importances_df, ave_SHAP_vlues_df

# ====================================================== End of CatBoost Code =================================================================== #

# =========================================================== Code for ANN forecast ================================================================== #
# Create an ANN model
def create_ANN_model(loss = closs.loss_As1_tf, # a loss function
                                        metrics = closs.As1_score_tf, # an evaluation score
                                        n_layers = 2, # the number of hidden layers
                                        hidden_layer_sizes = [100, 50], # the number of neurons in each hidden layer
                                        n_features = 10, # the number of features
                                        dropout = [0.4, 0.4], # the dropout rate in each hidden layer
                                        l1 = [0.01, 0.01], # the L1 penalty parameter for each hidden layer 
                                        l2 = [0.01, 0.01], # the L2 penalty parameter for each hidden layer
                                        ):
    model = Sequential()

    # define the input layer
    model.add( InputLayer(input_shape=(n_features, ), name='Input_layer') )

    # add a normalization
    model.add( Normalization() )

        # define hidden layers
    for i in np.arange(n_layers):
        n_neurons = hidden_layer_sizes[i]
        model.add( Dense(  units = n_neurons,
                                        activation = 'relu', 
                                        name = f'Hidden_layer_{i}',
                                        kernel_regularizer = regularizers.l1_l2(l1=l1[i], l2=l2[i])) )
        model.add( Dropout(rate = dropout[i]) )

    # define the output layer
    model.add( Dense(1, activation = 'sigmoid', name='Output_layer') )

    # compile the ANN model
    model.compile(loss = loss, optimizer = 'adam', metrics = metrics)  # default loss function: 'binary_crossentropy', default metrics: ['accuracy', AUC(name='auc')]
    # model.summary()
    return model

# Optimize the number of hidden layers, their neurons, the dropout rate in each hidden layer, and the degrees of L1 and L2 regularization in each hidden layer
def optimize_ANN_model(trial, n_features: int, loss = closs.loss_As1_tf, metrics = closs.As1_score_tf):

    n_layers = trial.suggest_int("n_layers", 1, 5)

    model = Sequential()

    # define the input layer
    model.add( InputLayer(input_shape=(n_features, ), name='Input_layer') )

    # add a normalization
    model.add( Normalization() )

    # define hidden layers
    for i in range(n_layers):
        n_neurons = trial.suggest_int("n_units_l{}".format(i), 10, 500, log=True)
        l1 = trial.suggest_float("L1_l{}".format(i), 0., 100)
        l2 = trial.suggest_float("L2_l{}".format(i), 0., 100)
        model.add( Dense(  units = n_neurons,
                                        activation = 'relu', 
                                        name = f'Hidden_layer_{i}',
                                        kernel_regularizer = regularizers.l1_l2(l1=l1, l2=l2)) )
        dropout = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.8)
        model.add( Dropout(rate = dropout) )

    # define the output layer
    model.add( Dense(1, activation = 'sigmoid', name = 'Output_layer') )

    # compile the ANN model
    model.compile(loss = loss, optimizer = 'adam', metrics = metrics)  # default loss function: 'binary_crossentropy', default metrics: ['accuracy', AUC(name='auc')]

    # model.summary()
    return model

# Define an objective function to be maximized with Optuna
def objective(  trial, # an Optuna object
                        X: np.array, # a numpy array of features
                        Y: np.array, # a numpy array of outcomes
                        cv, # a Sklearn data splitting scheme
                        loss = closs.loss_As1_tf, # a loss function
                        metrics = closs.As1_score_tf, # an evaluation metric
                        metrics_alias = 'As1_score_tf', # the string value of the evaluation metric
                        batch_size = 100, # the number of observations used by a stochastic optimization algorithm
                        num_epochs = 5, # the number of loops through the dataset used by a stochastic optimization algorithm
                        ):

    # Clear clutter from previous session graphs.
    K.clear_session()

    
    # Generate our trial model.
    model = optimize_ANN_model(trial, n_features = X.shape[1], loss = loss, metrics = metrics)

    cv_scores = []
    # Split the train data into train and validation subsets and iterate through them
    for _, (train_idx, valid_idx) in enumerate( cv.split(X, Y) ):
        # Split into training and validation CV sets
        X_train_cv, X_valid_cv = X[train_idx, :], X[valid_idx, :]
        y_train_cv, y_valid_cv = Y[train_idx], Y[valid_idx]

        # Fit the model on the training data.
        # The KerasPruningCallback checks for pruning condition every epoch.
        model.fit(
                        X_train_cv,
                        y_train_cv,
                        batch_size = batch_size,
                        callbacks = [TFKerasPruningCallback(trial, "val_{}".format(metrics_alias))],
                        epochs = num_epochs,
                        validation_data = (X_valid_cv, y_valid_cv),
                        verbose = 0)

        # Calculate the model loss and accuracy on the validation data
        score = model.evaluate(X_valid_cv, y_valid_cv)
        cv_scores.append(score[1])
    return np.mean(cv_scores)

# Calculate classfication scores of validation subsets
def class_scores(  trial, # an Optuna object
                            X: np.array, # a numpy array of features
                            Y: np.array, # a numpy array of outcomes
                            cv, # a Sklearn data splitting scheme
                            loss = closs.loss_As1_tf, # a loss function
                            metrics = closs.As1_score_tf, # an evaluation metric
                            metrics_alias = 'As1_score_tf', # the string value of the evaluation metric
                            batch_size = 100, # the number of observations used by a stochastic optimization algorithm
                            num_epochs = 5, # the number of loops through the dataset used by a stochastic optimization algorithm
                            ):
    """
        Calculate classification scores of validation data
    """
    # Clear clutter from previous session graphs.
    K.clear_session()

    
    # Generate our trial model.
    model = optimize_ANN_model(trial, n_features = X.shape[1], loss = loss, metrics = metrics)

    accuracy_scores, ave_precisions, precisions, aucs = [], [], [], []

    # Split the train data into train and validation subsets and iterate through them
    for _, (train_idx, valid_idx) in enumerate( cv.split(X, Y) ):
        # Split into training and validation CV sets
        X_train_cv, X_valid_cv = X[train_idx, :], X[valid_idx, :]
        y_train_cv, y_valid_cv = Y[train_idx], Y[valid_idx]

        # Fit the model on the training data.
        # The KerasPruningCallback checks for pruning condition every epoch.
        model.fit(
                        X_train_cv,
                        y_train_cv,
                        batch_size = batch_size,
                        callbacks = [TFKerasPruningCallback(trial, "val_{}".format(metrics_alias))],
                        epochs = num_epochs,
                        validation_data = (X_valid_cv, y_valid_cv),
                        verbose = 0)

        # Predict the validation subset
        y_valid_probs = model.predict(X_valid_cv, batch_size = batch_size, verbose = 0)
        y_valid_preds = (y_valid_probs > 0.5).astype('int')

        # Calculate classification scores
        accuracy_scores.append( accuracy_score(y_valid_cv, y_valid_preds) )
        ave_precisions.append( average_precision_score(y_valid_cv, y_valid_preds, average = 'micro') )
        precisions.append( precision_score(y_valid_cv, y_valid_preds, average = 'micro') )
        aucs.append(roc_auc_score(y_valid_cv, y_valid_probs) )

    return {'accuracy': np.mean(accuracy_scores), 'average_precision': np.mean(ave_precisions), 'precision': np.mean(precisions), 'auc': np.mean(aucs)}

# Forecast with ANN
def ANNf(   R: np.array, # an array of labels
                    X: np.array, # an array of features with columns as features
                    feat_names: list, # a list of feature names
                    tau: int, # a forecast horizon
                    batch_size: int, # the size of subsample used by a stochastic optimization algorithm
                    num_epochs: int, # the number of loops through the dataset used by a stochastic optimization algorithm
					loss = closs.loss_As1_tf, # a loss function
					metrics = closs.As1_score_tf, # an evaluation metric
                    metrics_alias = 'As1_score_tf', # the string value of the evaluation metric
                    n_trials = 100, # the number of trials set for Optuna
                    n_jobs = 1, # the number of workers for multiprocessing
                    ):
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    assert (tau > 0), "the forecast horizon (tau) must be an integer"
    T = X.shape[0]
    dim = X.shape[1]
    # R = R.flatten()
    # X = X.flatten() # flatten arrays

    R1 = np.empty( shape = (0, 1) )
    X1 = np.empty( shape = (0, dim) )
    for t in np.arange(0, T):
        if t < T-tau:
            R1 = np.append(R1, R[t+tau].reshape(1, 1), axis = 0)
        else:
            R1 = np.append(R1, np.array([0]).reshape(1, 1), axis = 0)
        X1 = np.append(X1, X[t, :].reshape(1, dim), axis = 0)

    # Rescale data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X1 = scaler.fit_transform(X1)
    # R1 = scaler.fit_transform(R1)

    # Studentize data
    # scaler = StandardScaler()
    # X1 = scaler.fit_transform(X1)
    # R1 = scaler.fit_transform(R1)

    # split the numpy array into train and test data
    if tau == 0:
        X1_train, X1_test = X1[0:(T-1), :], X1[(T-1):, :]
        R1_train, R1_test = R1[0:(T-1), :], R1[(T-1):, :]
    else:
        X1_train, X1_test = X1[0:(T-tau), :], X1[(T-tau):, :]
        R1_train, R1_test = R1[0:(T-tau)], R1[(T-tau):]

    # use the time-series cross-validation
    # tscv = TimeSeriesSplit(n_splits = 2, test_size = 10) 

    # use the stratified K-fold cross-validation
    skfcv = StratifiedKFold(n_splits = 5, shuffle=True, random_state = SEED)

    obj_func = lambda trial: objective(trial, X1_train, R1_train, cv = skfcv, loss = loss, metrics = metrics, metrics_alias = metrics_alias,
                                                                                                                                    batch_size = batch_size, num_epochs = num_epochs)

    # Create a new Optuna study
    sampler = RandomSampler(seed=SEED)
    study = optuna.create_study( direction = "maximize", # minimize a loss function
                                                    study_name = "ANN Classifier",
                                                    pruner = optuna.pruners.MedianPruner(),
                                                    sampler = sampler,
                                                    )

    # Perform cross validation
    study.optimize(obj_func, n_trials = n_trials)

    display( plot_parallel_coordinate(study) )
    
    trial = study.best_trial # get the best trial
    
    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    # print("  Value: ", trial.value)

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))

    # Refit the model using the optimal hyperparameter values
    best_model = optimize_ANN_model(trial, dim, loss = loss, metrics = metrics)
    best_model.fit(X1_train, R1_train, epochs=num_epochs, batch_size=batch_size, verbose=0) # Verbosity mode 0: silent

    # Forecast the test data
    forecasts = best_model.predict(X1_test)
    # print('forecasts = ', forecasts)
    forecasts = (forecasts > 0.5).astype('int')
    if tau > 1:
        forecast_tau = forecasts[len(forecasts)-1]
    else:
        forecast_tau = forecasts

    # Calculate the everage classification scores on the validation data
    ave_scores = class_scores(trial, X1_train, R1_train, cv = skfcv, loss = loss, metrics = metrics, metrics_alias = metrics_alias, batch_size = batch_size, num_epochs = num_epochs)

    # Create an ANN model using the optimal hyperparameters
    opt_parms_optuna = trial.params
    n_layers = opt_parms_optuna['n_layers']
    opt_params_sklearn = { 'loss': loss,
                                            'metrics': metrics,
                                            'n_layers': n_layers,
                                            'n_features': dim,
                                            'hidden_layer_sizes': [opt_parms_optuna[f'n_units_l{i}'] for i in range(n_layers)],
                                            'dropout': [opt_parms_optuna[f'dropout_l{i}'] for i in range(n_layers)],
                                            'l1': [opt_parms_optuna[f'L1_l{i}'] for i in range(n_layers)],
                                            'l2': [opt_parms_optuna[f'L2_l{i}'] for i in range(n_layers)]
                                        }

    print('opt_params_sklearn = ', opt_params_sklearn)
    best_model = KerasClassifier(build_fn = create_ANN_model, **opt_params_sklearn, epochs = num_epochs, batch_size = batch_size, verbose = 0)

    # Calculate average feature importance scores
    best_model.fit(X1_train, R1_train)
    permutation_imp = permutation_importance(best_model, X1_train, R1_train, n_jobs = 1, scoring = 'roc_auc', n_repeats = 8, random_state = SEED)
    features_importances_df = pd.DataFrame({'feature': feat_names, 'importance_score': permutation_imp.importances_mean}).round(decimals=2)\
                                                                                                                                                    .sort_values(by = 'importance_score', ascending=False)
    display(features_importances_df)

    # Calculate average SHAP values based on the best model fitted to the training sample
    X1_train_df = pd.DataFrame(X1_train, columns = feat_names)
    ave_SHAP_vlues = ave_SHAP_values(best_model, X1_train_df, use_method = 'deep_learning')
    display(ave_SHAP_vlues)

    K.clear_session()

    del best_model # delete all models
    gc.collect()

    # output results
    return int(forecast_tau), ave_scores, opt_params_sklearn

# ======================================================== End of ANN forecast ================================================================== #


# Create a LightGBM model
def create_LGBM_model(  objective = closs.update_As1_lgbm, # a default/custom loss function (cf. https://lightgbm.readthedocs.io/en/latest/Parameters.html)
                                            boosting_type = 'gbdt', # use the traditional Gradient Boosting Decision Tree (‘gbdt’), Dropouts meet Multiple Additive Regression Trees ('dart'),
                                                                                 # or Random Forest ('rf') which does not support custom objective functions
                                            max_depth = -1, # maximum depth of each decision tree (set it to a lower value to prevent overfitting)
                                            num_leaves = 31, # maximum number of leaves in each tree
                                            learning_rate = 1., # learning rate of the gradient descent optimization algorithm
                                            min_child_samples = 20, # minimum number of data in a leaf node (set it to a higher value to prevent overfitting)
                                            feature_fraction = 1., # subsample ratio of columns when constructing each tree
                                            bagging_fraction = 1., # subsample ratio of the training examples (set it to a lower value to prevent overfitting)
                                            n_estimators = 100, # number of gradient boosted trees
                                            ):
    model =  LGBMClassifier( objective = objective, boosting_type = boosting_type, max_depth = max_depth, num_leaves = num_leaves, learning_rate = learning_rate, \
                                                min_child_samples = min_child_samples, feature_fraction = feature_fraction, bagging_fraction = bagging_fraction, n_estimators = n_estimators, \
                                                force_row_wise = True, random_state = SEED, n_jobs = 2, verbosity = -100) 
    return model

# Forecast with LightGBM
def LGBMf(R: pd.DataFrame, # a dataframe of class labels
                    X: np.array, # a numpy array of features with columns as features
                    data_df: pd.DataFrame, # a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags
                    feat_names: list, # list of feature names
                    tau: int, # a forecast horizon
                    objective = closs.update_As1_lgbm, # an objective function
                    scoring_fn = 'AUC',  # a scoring function used to cross validate a LightGBM model 
                                                      # This argument can takes: ['Accuracy', 'Average_Precision', 'Precision', 'F1_score', 'AUC', 'Cross_entropy', 'As1_score', 'As2_score', 'Boost_score', 'Brier_score', 
                                                      # 'Gain_to_pain_ratio_fixed_trans_cost', 'Gain_to_pain_ratio_variable_trans_cost', 'Calmar_ratio_fixed_trans_cost', 'Calmar_ratio_variable_trans_cost', 
                                                      # 'Sharpe_ratio_fixed_trans_cost', 'Sharpe_ratio_variable_trans_cost', 'Sortino_ratio_fixed_trans_cost', 'Sortino_ratio_variable_trans_cost', 
                                                      # 'CECPP_fixed_trans_cost', 'CECPP_variable_trans_cost']
                    use_custom_loss = True, # whether or not to use a custom loss function
                    init_wealth = 1000, # an initial wealth
                    fixed_trans_cost = 10, # the dollar amount of fixed transaction cost 
                    variable_trans_cost = 0.005, # an amount of transaction cost as the percentage of stock price
                    n_trials = 100, # the number of trials set for Optuna
                    n_jobs = 1, # the number of workers for multiprocessing
                    ):
    ''' Forecast with LightGBM
    '''
    assert (R.shape[0] == X.shape[0]), "numbers of rows not match!"
    assert(tau > 0), "the forecast horizon must be greater than zero!"
    
    T = X.shape[0]
    dim = X.shape[1]
 
    R1 = R.shift(periods = -tau) # shift the labels forward

    # # Studentize data
    # scaler = StandardScaler()
    # X1 = scaler.fit_transform(X1)

    # split the numpy array into train and test data
    X1_train, X1_test = X[0:(T-tau), :], X[(T-tau):, :]
    R1_train, R1_test = R1.iloc[0: (T-tau)], R1.iloc[(T-tau):]

    # Rescale data
    scaler = MinMaxScaler( feature_range=(0, 1) )
    scaler.fit(X1_train)
    X1_train = scaler.transform(X1_train)
    X1_test = scaler.transform(X1_test)

    # Build a LightGBM model
    model = create_LGBM_model(objective = objective)

    # Define the grid search parameters
    if not use_custom_loss:
        objective = optuna.distributions.CategoricalDistribution(['binary', 'cross_entropy']) # set a loss function: 'cross_entropy', 'cross_entropy_lambda' (intensity-weighted), 
        # eval_metric = 'cross_entropy' # set a default evaluation metric for cross-validation: 'cross_entropy', 'cross_entropy_lambda', 'binary_logloss', 'auc'
    boosting_type = optuna.distributions.CategoricalDistribution(['gbdt', 'dart', 'goss']) # a tree model
    max_depth = optuna.distributions.IntDistribution(1, 6) # a maximum depth of each tree
    num_leaves = optuna.distributions.IntDistribution(10, 500, step = 10) # maximum number of leaves in each tree
    learning_rate = optuna.distributions.UniformDistribution(0., 100.) # a learning rate used for the Stochastic Gradient Descent
    min_child_samples = optuna.distributions.IntDistribution(10, 50) # minimum number of data in a leaf node (set it to a higher value to prevent overfitting)
    feature_fraction = optuna.distributions.UniformDistribution(0., 1.)  # subsample ratio of columns when constructing each tree
    bagging_fraction = optuna.distributions.UniformDistribution(0., 1.)  # subsample ratio of the training examples (set it to a lower value to prevent overfitting)
    n_estimators = optuna.distributions.IntDistribution(10, 200, step = 10) # a number of boosting iterations to be performed

    if use_custom_loss:
        # Define scoring functions (note the convention that higher score values are better than lower score values)
        gain_to_pain_ratio_score_fixed_trans_cost_lgbm = lambda model, X, y: closs.gain_to_pain_ratio_score_lgbm( model, X, y, 
                                                                                                                                                                                            data_df = data_df,
                                                                                                                                                                                            init_wealth = init_wealth, 
                                                                                                                                                                                            use_fixed_trans_cost = True,
                                                                                                                                                                                            fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                            variable_trans_cost = variable_trans_cost)
        gain_to_pain_ratio_score_variable_trans_cost_lgbm = lambda model, X, y: closs.gain_to_pain_ratio_score_lgbm(model, X, y, 
                                                                                                                                                                                                data_df = data_df,
                                                                                                                                                                                                init_wealth = init_wealth, 
                                                                                                                                                                                                use_fixed_trans_cost = False,
                                                                                                                                                                                                fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                                variable_trans_cost = variable_trans_cost)

        calmar_ratio_score_fixed_trans_cost_lgbm = lambda model, X, y: closs.calmar_ratio_score_lgbm( model, X, y, 
                                                                                                                                                                        data_df = data_df,
                                                                                                                                                                        init_wealth = init_wealth, 
                                                                                                                                                                        use_fixed_trans_cost = True,
                                                                                                                                                                        fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                        variable_trans_cost = variable_trans_cost)
        calmar_ratio_score_variable_trans_cost_lgbm = lambda model, X, y: closs.calmar_ratio_score_lgbm(model, X, y, 
                                                                                                                                                                            data_df = data_df,
                                                                                                                                                                            init_wealth = init_wealth, 
                                                                                                                                                                            use_fixed_trans_cost = False,
                                                                                                                                                                            fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                            variable_trans_cost = variable_trans_cost)

        sharpe_ratio_score_fixed_trans_cost_lgbm = lambda model, X, y: closs.sharpe_ratio_score_lgbm(  model, X, y, 
                                                                                                                                                                        data_df = data_df,
                                                                                                                                                                        init_wealth = init_wealth, 
                                                                                                                                                                        use_fixed_trans_cost = True,
                                                                                                                                                                        fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                        variable_trans_cost = variable_trans_cost)
        sharpe_ratio_score_variable_trans_cost_lgbm = lambda model, X, y: closs.sharpe_ratio_score_lgbm(  model, X, y, 
                                                                                                                                                                            data_df = data_df,
                                                                                                                                                                            init_wealth = init_wealth, 
                                                                                                                                                                            use_fixed_trans_cost = False,
                                                                                                                                                                            fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                            variable_trans_cost = variable_trans_cost)

        sortino_ratio_score_fixed_trans_cost_lgbm = lambda model, X, y: closs.sortino_ratio_score_lgbm( model, X, y, 
                                                                                                                                                                        data_df = data_df,
                                                                                                                                                                        init_wealth = init_wealth, 
                                                                                                                                                                        use_fixed_trans_cost = True,
                                                                                                                                                                        fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                        variable_trans_cost = variable_trans_cost)
        sortino_ratio_score_variable_trans_cost_lgbm = lambda model, X, y: closs.sortino_ratio_score_lgbm(model, X, y, 
                                                                                                                                                                            data_df = data_df,
                                                                                                                                                                            init_wealth = init_wealth, 
                                                                                                                                                                            use_fixed_trans_cost = False,
                                                                                                                                                                            fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                            variable_trans_cost = variable_trans_cost)

        cecpp_fixed_trans_cost_lgbm = lambda model, X, y: closs.cecpp_lgbm(model, X, y, 
                                                                                                                            data_df = data_df,
                                                                                                                            init_wealth = init_wealth, 
                                                                                                                            use_fixed_trans_cost = True,
                                                                                                                            fixed_trans_cost = fixed_trans_cost,
                                                                                                                            variable_trans_cost = variable_trans_cost)
        cecpp_variable_trans_cost_lgbm = lambda model, X, y: closs.cecpp_lgbm(model, X, y, 
                                                                                                                                data_df = data_df,
                                                                                                                                init_wealth = init_wealth, 
                                                                                                                                use_fixed_trans_cost = False,
                                                                                                                                fixed_trans_cost = fixed_trans_cost,
                                                                                                                                variable_trans_cost = variable_trans_cost)

        scoring = {'Accuracy': closs.accuracy_score_lgbm, 'Average_Precision': closs.average_precision_score_lgbm, 'Precision': closs.precision_score_lgbm, 'F1_score': closs.f1_score_lgbm, \
                            'AUC': closs.roc_auc_lgbm, 'Cross_entropy': closs.neg_log_loss_score_lgbm, 'As1_score': closs.As1_score_lgbm, 'As2_score': closs.As2_score_lgbm, \
                            'Boost_score': closs.Boost_score_lgbm, 'Brier_score': closs.Brier_score_lgbm, \
                            'Gain_to_pain_ratio_fixed_trans_cost': gain_to_pain_ratio_score_fixed_trans_cost_lgbm, \
                            'Gain_to_pain_ratio_variable_trans_cost': gain_to_pain_ratio_score_variable_trans_cost_lgbm, \
                            'Calmar_ratio_fixed_trans_cost': calmar_ratio_score_fixed_trans_cost_lgbm, 'Calmar_ratio_variable_trans_cost': calmar_ratio_score_variable_trans_cost_lgbm, \
                            'Sharpe_ratio_fixed_trans_cost': sharpe_ratio_score_fixed_trans_cost_lgbm, 'Sharpe_ratio_variable_trans_cost': sharpe_ratio_score_variable_trans_cost_lgbm, \
                            'Sortino_ratio_fixed_trans_cost': sortino_ratio_score_fixed_trans_cost_lgbm, 'Sortino_ratio_variable_trans_cost':  sortino_ratio_score_variable_trans_cost_lgbm, \
                            'CECPP_fixed_trans_cost': cecpp_fixed_trans_cost_lgbm, 'CECPP_variable_trans_cost': cecpp_variable_trans_cost_lgbm} 

        param_grid = dict(boosting_type = boosting_type, max_depth = max_depth, num_leaves = num_leaves, learning_rate = learning_rate, 
                                        min_child_samples = min_child_samples, feature_fraction = feature_fraction, bagging_fraction = bagging_fraction, 
                                        n_estimators = n_estimators)
    else:
        # List score functions used to train models. Note the convention that higher score values are better than lower score values
        gain_to_pain_ratio_score_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.gain_to_pain_ratio_score_sklearn(  y_true, 
                                                                                                                                                                                                        y_prob,
                                                                                                                                                                                                        data_df = data_df,
                                                                                                                                                                                                        init_wealth = init_wealth, 
                                                                                                                                                                                                        use_fixed_trans_cost = True, 
                                                                                                                                                                                                        fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                                        variable_trans_cost = variable_trans_cost)
        gain_to_pain_ratio_score_fixed_trans_cost_sklearn = make_scorer(gain_to_pain_ratio_score_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
        gain_to_pain_ratio_score_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.gain_to_pain_ratio_score_sklearn( y_true, 
                                                                                                                                                                                                            y_prob,
                                                                                                                                                                                                            data_df = data_df,
                                                                                                                                                                                                            init_wealth = init_wealth, 
                                                                                                                                                                                                            use_fixed_trans_cost = False, 
                                                                                                                                                                                                            fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                                            variable_trans_cost = variable_trans_cost)
        gain_to_pain_ratio_score_variable_trans_cost_sklearn = make_scorer(gain_to_pain_ratio_score_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

        calmar_ratio_score_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.calmar_ratio_score_sklearn(  y_true, 
                                                                                                                                                                                    y_prob,
                                                                                                                                                                                    data_df = data_df,
                                                                                                                                                                                    init_wealth = init_wealth, 
                                                                                                                                                                                    use_fixed_trans_cost = True, 
                                                                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                    variable_trans_cost = variable_trans_cost)
        calmar_ratio_score_fixed_trans_cost_sklearn = make_scorer(calmar_ratio_score_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
        calmar_ratio_score_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.calmar_ratio_score_sklearn( y_true, 
                                                                                                                                                                                        y_prob,
                                                                                                                                                                                        data_df = data_df,
                                                                                                                                                                                        init_wealth = init_wealth, 
                                                                                                                                                                                        use_fixed_trans_cost = False, 
                                                                                                                                                                                        fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                        variable_trans_cost = variable_trans_cost)
        calmar_ratio_score_variable_trans_cost_sklearn = make_scorer(calmar_ratio_score_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

        sharpe_ratio_score_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.sharpe_ratio_score_sklearn(  y_true, 
                                                                                                                                                                                    y_prob,
                                                                                                                                                                                    data_df = data_df,
                                                                                                                                                                                    init_wealth = init_wealth, 
                                                                                                                                                                                    use_fixed_trans_cost = True, 
                                                                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                    variable_trans_cost = variable_trans_cost)
        sharpe_ratio_score_fixed_trans_cost_sklearn = make_scorer(sharpe_ratio_score_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
        sharpe_ratio_score_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.sharpe_ratio_score_sklearn( y_true, 
                                                                                                                                                                                        y_prob,
                                                                                                                                                                                        data_df = data_df,
                                                                                                                                                                                        init_wealth = init_wealth, 
                                                                                                                                                                                        use_fixed_trans_cost = False, 
                                                                                                                                                                                        fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                        variable_trans_cost = variable_trans_cost)
        sharpe_ratio_score_variable_trans_cost_sklearn = make_scorer(sharpe_ratio_score_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

        sortino_ratio_score_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.sortino_ratio_score_sklearn(  y_true, 
                                                                                                                                                                                    y_prob,
                                                                                                                                                                                    data_df = data_df,
                                                                                                                                                                                    init_wealth = init_wealth, 
                                                                                                                                                                                    use_fixed_trans_cost = True, 
                                                                                                                                                                                    fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                    variable_trans_cost = variable_trans_cost)
        sortino_ratio_score_fixed_trans_cost_sklearn = make_scorer(sortino_ratio_score_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
        sortino_ratio_score_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.sortino_ratio_score_sklearn( y_true, 
                                                                                                                                                                                        y_prob,
                                                                                                                                                                                        data_df = data_df,
                                                                                                                                                                                        init_wealth = init_wealth, 
                                                                                                                                                                                        use_fixed_trans_cost = False, 
                                                                                                                                                                                        fixed_trans_cost = fixed_trans_cost,
                                                                                                                                                                                        variable_trans_cost = variable_trans_cost)
        sortino_ratio_score_variable_trans_cost_sklearn = make_scorer(sortino_ratio_score_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

        cecpp_fixed_trans_cost_sklearn = lambda y_true, y_prob: closs.cecpp_sklearn( y_true, 
                                                                                                                                        y_prob,
                                                                                                                                        data_df = data_df,
                                                                                                                                        init_wealth = init_wealth, 
                                                                                                                                        use_fixed_trans_cost = True, 
                                                                                                                                        fixed_trans_cost = fixed_trans_cost,
                                                                                                                                        variable_trans_cost = variable_trans_cost)
        cecpp_fixed_trans_cost_sklearn = make_scorer(cecpp_fixed_trans_cost_sklearn, greater_is_better = True, needs_proba=True)
        cecpp_variable_trans_cost_sklearn = lambda y_true, y_prob: closs.cecpp_sklearn(y_true, 
                                                                                                                                            y_prob,
                                                                                                                                            data_df = data_df,
                                                                                                                                            init_wealth = init_wealth, 
                                                                                                                                            use_fixed_trans_cost = False, 
                                                                                                                                            fixed_trans_cost = fixed_trans_cost,
                                                                                                                                            variable_trans_cost = variable_trans_cost)
        cecpp_variable_trans_cost_sklearn = make_scorer(cecpp_variable_trans_cost_sklearn, greater_is_better = True, needs_proba=True)

        scoring = {'Accuracy': 'accuracy', 'Average_Precision': 'average_precision', 'Precision': 'precision',  'F1_score': 'f1', 'AUC': 'roc_auc', 'Cross_entropy': 'neg_log_loss', \
                            'As1_score': closs.As1_score_sklearn, 'As2_score': closs.As2_score_sklearn, 'Boost_score': closs.Boost_score_sklearn, 'Brier_score': closs.Brier_score_sklearn,  \
                            'Gain_to_pain_ratio_fixed_trans_cost': gain_to_pain_ratio_score_fixed_trans_cost_sklearn, \
                            'Gain_to_pain_ratio_variable_trans_cost': gain_to_pain_ratio_score_variable_trans_cost_sklearn, \
                            'Calmar_ratio_fixed_trans_cost': calmar_ratio_score_fixed_trans_cost_sklearn, 'Calmar_ratio_variable_trans_cost': calmar_ratio_score_variable_trans_cost_sklearn, \
                            'Sharpe_ratio_fixed_trans_cost': sharpe_ratio_score_fixed_trans_cost_sklearn, 'Sharpe_ratio_variable_trans_cost': sharpe_ratio_score_variable_trans_cost_sklearn, \
                            'Sortino_ratio_fixed_trans_cost': sortino_ratio_score_fixed_trans_cost_sklearn, 'Sortino_ratio_variable_trans_cost': sortino_ratio_score_variable_trans_cost_sklearn, \
                            'CECPP_fixed_trans_cost': cecpp_fixed_trans_cost_sklearn, 'CECPP_variable_trans_cost': cecpp_variable_trans_cost_sklearn}  

        param_grid = dict(objective = objective, boosting_type = boosting_type, max_depth = max_depth, num_leaves = num_leaves, learning_rate = learning_rate, \
                                        min_child_samples = min_child_samples, feature_fraction = feature_fraction, bagging_fraction = bagging_fraction, n_estimators = n_estimators)


	# use time-series cross-validation
    # tscv = TimeSeriesSplit(n_splits = 2, test_size = 40) 

    # # use the stratified K-fold cross-validation
    skfcv = StratifiedKFold(n_splits = 5, shuffle=True, random_state = SEED)

    # Create a new Optuna study
    sampler = TPESampler(seed=SEED) # perform Bayesian optimization hyperparameter tuning
    # sampler = RandomSampler(seed=SEED) # perform Random Search
    study = optuna.create_study( direction = "maximize", # minimize a loss function
                                                    study_name = "LGBM Classifier",
                                                    pruner=optuna.pruners.HyperbandPruner(),
                                                    sampler = sampler)

	# Perform grid search
    model_cv = optuna.integration.OptunaSearchCV(model, param_distributions = param_grid, cv = skfcv,  study = study, scoring = scoring[scoring_fn], refit = True, n_jobs = n_jobs, n_trials=n_trials)
    
    # Cross-validate a model by using the grid search
    model_cv.fit( X1_train, R1_train)

    # # Plot the optimization history
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.update_layout(autosize=False, width=1100, height=600)
    # fig.update_xaxes(automargin=True)
    # display(fig)
    
    # # Generate the coordinate plot of the hyperparameter tuning process
    # fig = plot_parallel_coordinate(study)
    # fig.update_layout(autosize=False, width=1100, height=600)
    # fig.update_xaxes(automargin=True)
    # display(fig)

    # Retrieve the best model
    best_model = model_cv.best_estimator_

    # # Plot decision trees
    # fig, ax = plt.subplots( figsize=(10, 6) ) 
    # tree = lgbm.plot_tree(best_model, ax=ax, tree_index=1, figsize=(10, 6), dpi = 600, show_info = ['split_gain','leaf_count', 'internal_weight', 'leaf_weight'], \
    #                                                                                                                                                                                                              precision=2, orientation='horizontal')
    # display(tree)
    # fig.savefig( './tree.png',    # Set path and filename
    #                     dpi = 300,                     # Set dots per inch
    #                     bbox_inches="tight",           # Remove extra whitespace around plot
    #                     facecolor='white')             # Set background color to white

    # Forecast the test data
    if use_custom_loss:
        forecasts = sigmoid( model_cv.predict(X1_test) )
    else:
        forecasts = model_cv.predict_proba(X1_test)[:, 1]

    # forecasts = (forecasts > 0.5).astype('int')

    if tau > 1:
        forecast_tau = forecasts[len(forecasts)-1]
    else:
        forecast_tau = forecasts

    # Retrieve the optimal hyperparameter values
    opt_params = model_cv.best_params_

    # Calculate the feature importance scores based on the best model fitted to the training sample
    features_importances_df = pd.DataFrame({'feature': feat_names, 'importance_score': best_model.feature_importances_}).round(decimals=2) \
                                                                                                                                                    #.sort_values(by = 'importance_score', ascending=False)
    # display(features_importances_df)

    # Calculate average SHAP values based on the best model fitted to the training sample
    X1_train_df = pd.DataFrame(X1_train, columns = feat_names)
    ave_SHAP_vlues_df = ave_SHAP_values(best_model, X1_train_df)
    # display(ave_SHAP_vlues_df)

    # permutation_imp_df, importance_boxplot = box_plot_per_feats(best_model, X1_train_df, R1_train, scoring = 'roc_auc')
    # display(permutation_imp_df)

    # Calculate the classification scores for the validation data
    scores = cross_validate(best_model, X1_train, R1_train, cv = skfcv, scoring = scoring, n_jobs = 10)
    scores_median = {key: np.median(v[~np.isnan(v)]) for key, v in scores.items()}

    del model, model_cv, best_model # delete all models
    gc.collect()

    # output results
    return float(forecast_tau), scores_median, opt_params, features_importances_df, ave_SHAP_vlues_df


# Generate toy coordinate plot and tree plot
def LGBM_plot( R_df: pd.DataFrame, # a pandas dataframe of class labels
                            X_df: pd.DataFrame, # a pandas dataframe of features with columns as features
                            tau: int, # a forecast horizon
                            objective = closs.update_As1_lgbm, # an objective function
                            eval_metric = closs.As1_err_lgbm, # an evaluation metric
                            scoring_fn = 'AUC',  # a scoring function used to rank models during the cross validation. 
                                                            # This argument can takes: 'As1_score',  'As2_score', 'Boost_score', 'Brier_score'
                            use_custom_loss = True, # whether or not to use a custom loss function
                            n_trials = 100, # the number of trials set for Optuna
                            n_jobs = 1, # the number of workers for multiprocessing
                            ):
    """ Generate toy coordinate plot and tree plot
    """
    # Rescale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_df)
    X1_df = pd.DataFrame( scaler.transform(X_df), columns = X_df.columns.to_list() )

    # Build a LightGBM model
    model = create_LGBM_model(objective = objective)

    # Define the grid search parameters
    if not use_custom_loss:
        objective = optuna.distributions.CategoricalDistribution(['binary', 'cross_entropy']) # set a loss function: 'cross_entropy', 'cross_entropy_lambda' (intensity-weighted), 
        eval_metric = 'cross_entropy' # set an evaluation metric for cross-validation: 'cross_entropy', 'cross_entropy_lambda', 'binary_logloss', 'auc'
    boosting_type = optuna.distributions.CategoricalDistribution(['gbdt', 'dart', 'goss']) # set a boosting type: gbdt (traditional Gradient Boosting Decision Tree)
                                                                                                                                        # rf (Random Forest)
                                                                                                                                        # dart (Dropouts meet Multiple Additive Regression Trees)
                                                                                                                                        # goss Gradient-based One-Side Sampling)
    learning_rate = optuna.distributions.UniformDistribution(0.01, 100) #the learning rate of the gradient descent optimization algorithm
    n_estimators = optuna.distributions.IntDistribution(10, 100) # set the number of boosted trees to fit
    max_depth = optuna.distributions.IntDistribution(1, 12) # set the maximum depth of each decision tree
    num_leaves = optuna.distributions.IntDistribution(100, 2000) # set the maximum number of leaves in each tree
    min_child_samples = optuna.distributions.IntDistribution(10, 100) # set the minimum number of samples required in the leaf nodes
    feature_fraction = optuna.distributions.UniformDistribution(0.1, 1.0) # set the fraction of randomly sampled features used to choose the best splitting point for each decision tree
    bagging_fraction = optuna.distributions.UniformDistribution(0.1, 1.0) # set the fraction of randomly sampled observations used to train each tree

    # List score functions used to train models. Note the convention that higher score values are better than lower score values
    if use_custom_loss:
        scoring = {'Accuracy': closs.accuracy_score_lgbm, 'Average_Precision': closs.average_precision_score_lgbm, 'Precision': closs.precision_score_lgbm, 'F1_score': closs.f1_score_lgbm, \
                            'AUC': closs.roc_auc_lgbm, 'As1_score': closs.As1_score_lgbm, 'As2_score': closs.As2_score_lgbm, 'Boost_score': closs.Boost_score_lgbm, 'Brier_score': closs.Brier_score_lgbm} 
        param_grid = dict(boosting_type = boosting_type, learning_rate = learning_rate, n_estimators = n_estimators, max_depth = max_depth, num_leaves = num_leaves, \
                                                                                                                                    min_child_samples = min_child_samples, feature_fraction = feature_fraction, bagging_fraction = bagging_fraction)
    else:
        # List score functions used to train models. Note the convention that higher score values are better than lower score values
        scoring = {'Accuracy': 'accuracy', 'Average Precision': 'average_precision', 'Precision': 'precision', 'AUC': 'roc_auc', 'As1_score': closs.As1_score_sklearn, 'As2_score': closs.As2_score_sklearn, \
                                                                                                                                                                                    'Boost_score': closs.Boost_score_sklearn, 'Brier_score': closs.Brier_score_sklearn}  
        param_grid = dict(objective = objective, boosting_type = boosting_type, learning_rate = learning_rate, n_estimators = n_estimators, max_depth = max_depth, num_leaves = num_leaves, \
                                                                                                                                    min_child_samples = min_child_samples, feature_fraction = feature_fraction, bagging_fraction = bagging_fraction)

	# use time-series cross-validation
    # tscv = TimeSeriesSplit(n_splits = 2, test_size = 40) 

    # # use the stratified K-fold cross-validation
    skfcv = StratifiedKFold(n_splits = 5, shuffle=True, random_state = SEED)

    # Create a new Optuna study
    sampler = TPESampler(seed=SEED) # perform Bayesian optimization hyperparameter tuning
    # sampler = RandomSampler(seed=SEED) # perform Random Search
    study = optuna.create_study(direction = "maximize", # minimize a loss function
                                                    study_name = "LGBM Classifier",
                                                    pruner=optuna.pruners.HyperbandPruner(),
                                                    sampler = sampler)

	# Perform grid search
    if use_custom_loss:
        model_cv = optuna.integration.OptunaSearchCV(model, param_distributions = param_grid, cv = skfcv, study = study, \
                                                                                                                                                                    scoring = scoring[scoring_fn], refit = True, n_jobs = n_jobs, n_trials=n_trials)
    else:
        model_cv = optuna.integration.OptunaSearchCV(model, param_distributions = param_grid, cv = skfcv,  study = study, \
                                                                                                                                                                    scoring = scoring[scoring_fn], refit = True, n_jobs = n_jobs, n_trials=n_trials)

    # Cross-validate a model by using the grid search
    model_cv.fit(X1_df, R_df.values.ravel(), eval_metric = eval_metric)

    # if use_custom_loss:
    #     R1_pred =  sigmoid( model_cv.predict(X1_train) )
    # else:
    #     R1_pred =  model_cv.predict_proba(X1_train)[:, 1]
    # print('R1_pred = ', R1_pred)
    # R1_err = np.c_[R1_train.astype(int), (R1_pred > 0.5).astype('int')]
    # np.savetxt('./R1_error.csv', R1_err, delimiter = ' , ')
    
    # Generate the coordinate plot of the hyperparameter tuning process
    fig = plot_parallel_coordinate(study)
    fig.update_layout(autosize=False, width=1100, height=600)
    fig.update_xaxes(automargin=True)
    display(fig)

    # Retrieve the best model
    best_model = model_cv.best_estimator_

    # Plot decision trees
    fig, ax = plt.subplots( figsize=(10, 6) ) 
    tree = lgbm.plot_tree(best_model, ax=ax, tree_index=0, figsize = (10, 6), dpi = 600, show_info = ['split_gain'], precision=2, orientation='horizontal')
    display(tree)
    fig.savefig( './tree.pdf',    # Set path and filename
                        dpi = 600,                     # Set dots per inch
                        bbox_inches="tight",           # Remove extra whitespace around plot
                        facecolor='white')             # Set background color to white

    return True
# ====================================================== End of LightGBM Code =================================================================== #

