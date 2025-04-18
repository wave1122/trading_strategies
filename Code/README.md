# About The Project

This repository provides Python codes used to perform all calculations reported in the paper: `Technical Analysis with Machine Learning Classification Algorithms: Can it still ‘Beat’ the Buy-and-Hold Strategy?`

# Pre-requisites

* [Anaconda Distribution with Python 3.9.21](https://www.anaconda.com/)

# Calculate rolling-window forecasts

For example, the following script will calculate four-period-ahead forecasts using the rolling-window strategy for BTC-USD using *Dataset I* as predictors

```sh
python ./RF_BTC_all_vars_CE_tau_4.py > /dev/null 2>&1 &
```

# Implement a Trading Strategy

For example, the following script will implement the proposed trading strategy based on Random Forest forecasts and all the performance metrics for BTC-USD using *Dataset I* as predictors

```sh
python ./perf_report_RF_BTC_all_vars_v1.py > /dev/null 2>&1 &
```

# Calculate Statistics of each Performance Metrics

For example, the following script will calculate the medians, IQRs, means, standard deviations, t-statistics of performance metrics of the trading strategy based on Random Forest forecasts for BTC-USD

```sh
python ./perf_stats_all_invest_pers_v1.py > /dev/null 2>&1 &
```

# Implement Bootstrap Reality Check

For example, the following script will compare all trading methods based on LightGBM forecasts for SPY

```sh
python ./bootstrap_RC.py
```

# Plot Figures

For example, the following Jupyter notebook will plot all the figures in the paper

```py
plots_v1.ipynb
```



# List of Main Files

| Python main file                          | Description                                                  |
| ----------------------------------------- | ------------------------------------------------------------ |
| trading_strategies.py                     | Python functions to implement the proposed trading strategies |
| talib_technical_indicators.py             | Python functions to calculate technical indicators and candlestick chart patterns |
| performance_measures.py                   | Python functions to calculate performance measures           |
| custom_losses_v1.py                       | Python functions to define custom loss functions and scoring functions |
| compare_models.py                         | Python function to implement the bootstrap reality check to compare trading methods |
| forecast_directional_movement_v4.py       | Python function to implement the rolling-window strategy to predict future pricing moving directions |
| my_classification_algorithms_optuna_v4.py | Python functions to implement and cross validate all ML classification algorithms |

# License

Distributed under the MIT License. See `LICENSE.txt` for more information.

# Contact

Ba Chu -  [ba.chu@carleton.ca](mailto:ba.chu@carleton.ca)

Project Link: [https://github.com/wave1122/trading_strategies](https://github.com/wave1122/trading_strategies)
