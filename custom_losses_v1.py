import pandas as pd
import numpy as np
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
        
import gc
import math
from varname import nameof

# import a tupling module
from typing import Tuple

# import SciPy modules
from scipy.misc import derivative
from scipy.special import softmax

# import Sklearn evaluation metrics/scores
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, average_precision_score
from sklearn.metrics import make_scorer

# import Tensorflow API
import tensorflow as tf
from tensorflow.python.keras import backend as K

# import trading strategies and their performance measures
import trading_strategies as tt
import performance_measures as pf


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# # Import data (used to calculate the performance metrics of trading strategies)
# tau = 1 # this is the forecast horizon (which must be the same as the values set elwhere!)
# # data_df = pd.read_csv('../Data/SPY_all_vars.csv', encoding='utf-8', sep = ',', low_memory=False, header = 0, skiprows = 0, skipinitialspace=True, parse_dates = ['date'])
# data_df = pd.read_csv('../Data/RW_all_vars.csv', encoding='utf-8', sep = ',', low_memory=False, header = 0, skiprows = 0, skipinitialspace=True, parse_dates = ['date'])

# # Select the columns: 'date', 'price', and 'RF', then shift the last two columns forward 'tau' lags
# data_df = data_df[['date', 'price', 'RF']]
# data_df[['price', 'RF']] = data_df[['price', 'RF']].shift(periods = -tau)

# Define the sigmoid function
def sigmoid(x): return 1. / (1. +  np.exp(-x))

# ================================================== Define the custom loss functions and evaluation metrics for XGBoost ========================================== #

# ------------------------------------------------------------------------------------ Use As1 loss function to heavily penalize false negatives ------------------------------------------------------------------------------------ #
def update_As1_xgb(dtrain, y_predt) -> Tuple[np.ndarray, np.ndarray]: # This does not seem to work. Check it later
    """
        INPUT
            dtrain: training labels
            y_predt: values of the inverse logit function
    """
    # print('y_predt = ', y_predt)
    # print('dtrain = ', dtrain)
    y_true = dtrain.get_label() if isinstance(dtrain,xgb.DMatrix) else dtrain
    def As1_loss(x, t):
        p = sigmoid(x)
        p = np.clip(p, 10e-7, 1-10e-7)
        return -(t*(np.log(p) - p + 1.) - (1. - t)*p)

    loss = lambda x: As1_loss(x, y_true)
    gradient = derivative(loss, y_predt, n = 1, dx = 1e-6)
    hessian = derivative(loss, y_predt, n = 2, dx = 1e-6)
    return gradient, hessian

# Define the As1 evaluation metric
def As1_err_xgb(predt, dmat) -> Tuple[str, float]:
    """ Compute the evaluation function associated with the As1 loss function
    """
    y = dmat.get_label() if isinstance(dmat,xgb.DMatrix) else dmat
    p = sigmoid(predt)
    p = np.clip(p, 10e-7, 1-10e-7)
    loss_fn = -y * (np.log(p) - p + 1.)
    loss_fp = -(1. - y) * p
    return 'A1_error', np.mean(loss_fn - loss_fp)

# ----------------------------------------------------------------------- Use As2 loss function to heavily penalize false positives ---------------------------------------------------------------------------------------------------- #
def update_As2_xgb(dtrain, y_predt) -> Tuple[np.array, np.array]: 
    """
        INPUT
            dtrain: training labels
            y_predt: values of the inverse logit function
    """
    y_true = dtrain

    print('y_predt = ', y_predt)
    print('dtrain = ', dtrain)

    def As2_loss(x, t):
        p = sigmoid(x)
        p = np.clip(p, 10e-4, 1-10e-4)
        return -( t * (p - 1.) + (1. - t) * (p + np.log(1. - p)) )

    loss = lambda x: As2_loss(x, y_true)
    gradient = derivative(loss, y_predt, n = 1, dx = 1e-10)
    hessian = derivative(loss, y_predt, n = 2, dx = 1e-10)
    return gradient, hessian

# Define the As2 evaluation metric
def As2_err_xgb(predt, dmat) -> Tuple[str, float]:
    """ Compute the evaluation function associated with the As2 loss function
    """
    y = dmat
    p = sigmoid(predt)
    p = np.clip(p, 10e-7, 1-10e-7)
    loss_fn = -y * (p - 1.)
    loss_fp = -(1. - y) * (p + np.log(1. - p))
    return 'A2_error', np.mean(loss_fn + loss_fp)

# ------------------------------------------------------------------------ Define score functions to be used with the custom loss functions --------------------------------------------------------------------------------------------- #

# Define the custom As1 score function
def As1_score_xgb(model, X, y) -> np.float32:
    """ Compute the As1 score function
    """
    p = model.predict(X)[:, 1]
    p = np.clip(p, 10e-7, 1-10e-7)
    score_fn = y * (np.log(p) - p + 1)
    score_fp = (1. - y) * p
    return np.mean(score_fn - score_fp)

# ================================================= Finish defining the custom loss functions and evaluation metrics for XGBoost ===================================== #

# ================================================== Define the custom loss functions and evaluation metrics for CatBoost ========================================== #

# --------------------------------------------------------------------------------------- Use the standard log loss function -------------------------------------------------------------------------------------------------------- #
# Define the LogLoss function
class LoglossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            e = np.exp(approxes[index])
            p = e / (1 + e)
            der1 = targets[index] - p
            der2 = -p * (1 - p)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result

# Define the LogLoss metric
class LoglossMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            e = np.exp(approx[i])
            p = e / (1 + e)
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += -w * (target[i] * np.log(p) + (1 - target[i]) * np.log(1 - p))

        return error_sum, weight_sum

# ----------------------------------------------------------------------------------- Use As1 loss function to heavily penalize false negatives ---------------------------------------------------------------------------------- #
# Define the As1 loss function
class As1lossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            e = np.exp(approxes[index])
            p = e / (1 + e)
            p_der1 = p * (1. - p)
            p_der2 = p_der1 * (1. - 2*p)
            der1 = p_der1 * (targets[index]/p - 1.) 
            der2 =  (targets[index]/p - 1.) * p_der2 - (targets[index] / np.power(p, 2.)) * np.power(p_der1, 2.)
            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result

class As1lossObjective1(object):

    def As1_loss(self, x, t):
        prob = sigmoid(x)
        prob = np.clip(prob, 10e-7, 1-10e-7)
        return -(t*(np.log(prob) - prob + 1.) - (1. - t)*prob)

    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            loss = lambda x: self.As1_loss(x, targets[index])
            der1 = derivative(loss, approxes[index], n = 1, dx = 1e-6)
            der2 = derivative(loss, approxes[index], n = 2, dx = 1e-6)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result


# Define the As1 metric
class As1Metric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        # Returns whether great values of metric are better
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            e = np.exp(approx[i])
            p = e / (1 + e)
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            score_fn = target[i] * (np.log(p) - p + 1.)
            score_fp = (1. - target[i]) * p
            error_sum += -w * (score_fn - score_fp)

        return error_sum, weight_sum

# ----------------------------------------------------------------------------------- Use As2 loss function to heavily penalize false positives ---------------------------------------------------------------------------------- #
# Define the As2 loss function
class As2lossObjective(object):

    def As2_loss(self, x, t):
        prob = sigmoid(x)
        prob = np.clip(prob, 10e-7, 1-10e-7)
        return -( t * (prob - 1.) + (1. - t) * (prob + np.log(1. - prob)) )

    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            loss = lambda x: self.As2_loss(x, targets[index])
            der1 = derivative(loss, approxes[index], n = 1, dx = 1e-6)
            der2 = derivative(loss, approxes[index], n = 2, dx = 1e-6)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result

# Define the As2 metric
class As2Metric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        # Returns whether great values of metric are better
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            e = np.exp(approx[i])
            p = e / (1 + e)
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            score_fn = target[i] * (p - 1.)
            score_fp = (1. - target[i]) * (p + np.log(1. - p))
            error_sum += -w * (score_fn + score_fp)

        return error_sum, weight_sum
# ---------------------------------------------------------------------------------------- Use the boosting loss function of Buja et al. (2005) ---------------------------------------------------------------------------------------- #
# Define the boosting loss function
class BoostlossObjective(object):

    def boost_loss(self, x, t):
        prob = sigmoid(x)
        prob = np.clip(prob, 10e-7, 1-10e-7)
        return t * np.sqrt((1. - prob) / prob) + (1 - t) * np.sqrt(prob / (1. - prob))

    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            loss = lambda x: self.boost_loss(x, targets[index])
            der1 = derivative(loss, approxes[index], n = 1, dx = 1e-6)
            der2 = derivative(loss, approxes[index], n = 2, dx = 1e-6)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result

# Define the Boost metric
class BoostMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        # Returns whether great values of metric are better
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            e = np.exp(approx[i])
            p = e / (1 + e)
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            score_fn = target[i] * np.sqrt( (1. - p) / (p + 1e-6) )
            score_fp = (1. - target[i]) * np.sqrt( p / (1. - p + 1e-6) )
            error_sum += w * (score_fn + score_fp)

        return error_sum, weight_sum

# ---------------------------------------------------------------------------------------- Use the Brier loss function of Brier (1950) ---------------------------------------------------------------------------------------- #
# Define the Brier loss function
class BrierlossObjective(object):

    def Brier_loss(self, x, t):
        prob = sigmoid(x)
        prob = np.clip(prob, 10e-7, 1-10e-7)
        return t * np.power(1. - prob, 2.) + (1 - t) * np.power(prob, 2.)

    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            loss = lambda x: self.Brier_loss(x, targets[index])
            der1 = derivative(loss, approxes[index], n = 1, dx = 1e-6)
            der2 = derivative(loss, approxes[index], n = 2, dx = 1e-6)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result

# Define the Brier metric
class BrierMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        # Returns whether great values of metric are better
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            e = np.exp(approx[i])
            p = e / (1 + e)
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            score_fn = target[i] * np.power(1. - p, 2.)
            score_fp = (1. - target[i]) * np.power(p, 2.)
            error_sum += w * (score_fn + score_fp)

        return error_sum, weight_sum

# ---------------------------------------------------------------------------------------- Use various score functions to be used with custom loss functions  ---------------------------------------------------------------------- #
# Define the binary logloss score function
def accuracy_score_cb(model, X, y) -> np.float32:
    """ Compute the accuracy score
    """
    p = model.predict_proba(X, verbose = 0)[:, 1]
    p = np.clip(p, 10e-7, 1-10e-7)
    y_pred_binary = (p > 0.5).astype('int')
    return accuracy_score(y, y_pred_binary)

# Define the precision score
def precision_score_cb(model, X, y) -> np.float32:
    """ Compute the precision score
    """
    p = model.predict_proba(X, verbose = 0)[:, 1]
    p = np.clip(p,10e-7, 1-10e-7)
    y_pred_binary = (p > 0.5).astype('int')
    return precision_score(y, y_pred_binary, average = 'micro')

def average_precision_score_cb(model, X, y) -> np.float32:
    """ Compute average precision score
    """
    p = model.predict_proba(X, verbose = 0)[:, 1]
    p = np.clip(p, 10e-7, 1-10e-7)
    y_pred_binary = (p > 0.5).astype('int')
    return average_precision_score(y, y_pred_binary, average = 'micro')

# Define the AUC score
def roc_auc_cb(model, X, y) -> np.float32:
    """ Compute the AUC score
    """
    p = model.predict_proba(X, verbose = 0)[:, 1]
    p = np.clip(p, 10e-7, 1-10e-7)
    return roc_auc_score(y, p)

# Define the F1 score
def f1_score_cb(model, X, y) -> np.float32:
    """ Compute the F1 score
    """
    p = model.predict_proba(X, verbose = 0)[:, 1]
    p = np.clip(p, 10e-7, 1-10e-7)
    y_pred_binary = (p > 0.5).astype('int')
    return f1_score(y, y_pred_binary, average = 'micro')

# Define the As1 score function
def As1_score_cb(model, X, y) -> np.float32:
    """ Compute the As1 score
    """
    p = model.predict_proba(X, verbose = 0)[:, 1]
    p = np.clip(p, 10e-7, 1-10e-7)
    score_fn = y * (np.log(p) - p + 1)
    score_fp = (1. - y) * p
    return np.mean(score_fn - score_fp)

# Define the As2 score function
def As2_score_cb(model, X, y) -> np.float32:
    """ Compute the As2 score
    """
    p = model.predict_proba(X, verbose = 0)[:, 1]
    p = np.clip(p, 10e-7, 1-10e-7)
    score_fn = y * (p - 1.)
    score_fp = (1. - y) * (p + np.log(1. - p))
    return np.mean(score_fn + score_fp)

# Define the boosting score function
def Boost_score_cb(model, X, y) -> np.float32:
    """ Compute the boost score of Buja et al. (2005)
    """
    p = model.predict_proba(X, verbose = 0)[:, 1]
    p = np.clip(p, 10e-7, 1-10e-7)
    score_fn = -y * np.sqrt( (1. - p) / p )
    score_fp = -(1. - y) * np.sqrt( p / (1. - p) )
    return np.mean(score_fn + score_fp)

# Define the Brier score function
def Brier_score_cb(model, X, y) -> np.float32:
    """ Compute the Brier score function
    """
    p = model.predict_proba(X, verbose = 0)[:, 1]
    p = np.clip(p, 10e-7, 1-10e-7)
    score_fn = -y * np.power(1. - p, 2.)
    score_fp = -(1. - y) * np.power(p, 2.)
    return np.mean(score_fn + score_fp)

# ================================================= Finish defining the custom loss functions and evaluation metrics for CatBoost ====================================== #

# ================================================== Define the custom loss functions and score functions for ANN ================================================ #
# ----------------------------------------------------------------------------------- Use As1 loss function to heavily penalize false negatives ----------------------------------------------------------------------------------------- #
# Define the As1 loss function
def loss_As1_tf(  y_true, # training labels
                            predt): # values of the logit function
    """ Compute the custom As1 loss
    """
    y_true = tf.cast(y_true, tf.float32)
    # K.print_tensor(y_true, message='y_true = ')
    # K.print_tensor(predt, message='y_pred = ')

    losses = -( tf.math.multiply(y_true, tf.math.log(predt + 1E-6) - predt + 1.) - tf.math.multiply(1. - y_true, predt) )
    average_loss = tf.math.reduce_mean(losses)
    # K.print_tensor(average_loss, message='average_loss = ')
    return average_loss

# Define the As1 score function
def As1_score_tf(y_true, # training labels
                            predt): # values of the logit function
    """ Compute the custom As1 score function
    """
    # y_true = tf.cast(y_true, tf.float32)
    # K.print_tensor(y_true, message='y_true = ')
    # K.print_tensor(predt, message='y_pred = ')

    scores = tf.math.multiply(y_true, tf.math.log(predt + 1E-6) - predt + 1.) - tf.math.multiply(1. - y_true, predt) 
    average_score = tf.math.reduce_mean(scores)
    # K.print_tensor(average_score, message='average_score = ')
    return average_score

    # ----------------------------------------------------------------------------------- Use As2 loss function to heavily penalize false positives ----------------------------------------------------------------------------------------- #
# Define the As2 loss function
def loss_As2_tf(  y_true, # training labels
                            predt): # values of the logit function
    """ 
        Compute the custom As2 loss
    """
    y_true = tf.cast(y_true, tf.float32)
    # K.print_tensor(y_true, message='y_true = ')
    # K.print_tensor(predt, message='y_pred = ')

    losses = -( tf.math.multiply(y_true, predt - 1.) + tf.math.multiply( 1. - y_true, predt + tf.math.log(1. - predt) ) )
    average_loss = tf.math.reduce_mean(losses)
    # K.print_tensor(average_loss, message='average_loss = ')
    return average_loss

# Define the As2 score function
def As2_score_tf(y_true, # training labels
                            predt): # values of the logit function
    """ Compute the custom As1 score function
    """
    # y_true = tf.cast(y_true, tf.float32)
    # K.print_tensor(y_true, message='y_true = ')
    # K.print_tensor(predt, message='y_pred = ')

    scores = tf.math.multiply(y_true, predt - 1.) + tf.math.multiply( 1. - y_true, predt + tf.math.log(1. - predt) ) 
    average_score = tf.math.reduce_mean(scores)
    # K.print_tensor(average_score, message='average_score = ')
    return average_score

# ------------------------------------------------------------------------------- Use the boosting loss function of  Buja et al. (2005) --------------------------------------------------------------------------------------------------- #
# Define the boosting loss function
def loss_Boosting_tf(  y_true, # training labels
                            predt): # values of the logit function
    """ 
        Compute the custom boosting loss of Buja et al. (2005)
    """
    y_true = tf.cast(y_true, tf.float32)
    # K.print_tensor(y_true, message='y_true = ')
    # K.print_tensor(predt, message='y_pred = ')

    losses = tf.math.multiply( y_true, tf.math.sqrt( (1. - predt) / predt ) ) + tf.math.multiply( 1. - y_true, tf.math.sqrt( predt / (1. - predt) )  ) 
    average_loss = tf.math.reduce_mean(losses)
    # K.print_tensor(average_loss, message='average_loss = ')
    return average_loss

# Define the boosting score function
def Boosting_score_tf(y_true, # training labels
                            predt): # values of the logit function
    """ Compute the custom boosting score function
    """
    # y_true = tf.cast(y_true, tf.float32)
    # K.print_tensor(y_true, message='y_true = ')
    # K.print_tensor(predt, message='y_pred = ')

    scores = -( tf.math.multiply( y_true, tf.math.sqrt( (1. - predt) / predt ) ) + tf.math.multiply( 1. - y_true, tf.math.sqrt( predt / (1. - predt) )  ) )
    average_score = tf.math.reduce_mean(scores)
    # K.print_tensor(average_score, message='average_score = ')
    return average_score

# ---------------------------------------------------------------------------------------------------- Use the Brier loss function ------------------------------------------------------------------------------------------------------------ #
# Define the Brier loss function
def loss_Brier_tf(  y_true, # training labels
                                    predt): # values of the logit function
    """ 
        Compute the Brier loss function
    """
    y_true = tf.cast(y_true, tf.float32)
    # K.print_tensor(y_true, message='y_true = ')
    # K.print_tensor(predt, message='y_pred = ')

    losses = tf.math.multiply( y_true, tf.math.pow(1. - predt, 2.) ) + tf.math.multiply( 1. - y_true, tf.math.pow(predt, 2.)  ) 
    average_loss = tf.math.reduce_mean(losses)

    # K.print_tensor(average_loss, message='average_loss = ')
    return average_loss

# Define the Brier score function
def Brier_score_tf(y_true, # training labels
                            predt): # values of the logit function
    """ Compute the Brier score function
    """
    # y_true = tf.cast(y_true, tf.float32)
    # K.print_tensor(y_true, message='y_true = ')
    # K.print_tensor(predt, message='y_pred = ')

    scores = -( tf.math.multiply( y_true, tf.math.pow(1. - predt, 2.) ) + tf.math.multiply( 1. - y_true, tf.math.pow(predt, 2.)  )  )
    average_score = tf.math.reduce_mean(scores)
    # K.print_tensor(average_score, message='average_score = ')
    return average_score

# ================================================ Finish defining the custom loss functions and score functions for ANN ============================================= #

# =================================================== Define the custom loss functions and evaluation metrics for LightGBM ========================================= #
# ------------------------------------------------------------------------------------------- Use As1 loss function to heavily penalize false negatives ---------------------------------------------------------------------------------- #
# Define the As1 loss function
def update_As1_lgbm(y_true, y_predt) -> Tuple[np.ndarray, np.ndarray]:
    def As1_loss(x, label):
        p = sigmoid(x)
        return -(label*(np.log(p) - p + 1.) - (1. - label)*p)

    loss = lambda x: As1_loss(x, y_true)
    gradient = derivative(loss, y_predt, n = 1, dx = 1e-6)
    hessian = derivative(loss, y_predt, n = 2, dx = 1e-6)
    return gradient, hessian

# Define the As1 evaluation metric
def As1_err_lgbm(y_true, y_hat):
    """ Compute the evaluation function associated with the As1 loss function
    """
    prob = sigmoid(y_hat)
    prob = np.clip(prob,10e-7, 1-10e-7)
    score_fn = y_true * (np.log(prob) - prob + 1.)
    score_fp = (1. - y_true) * prob
    return  'As1_err_lgbm', np.mean(-(score_fn - score_fp)), False

# ------------------------------------------------------------------------------------------- Use As2 loss function to heavily penalize false positives ---------------------------------------------------------------------------------- #
# Define the As2 loss function
def update_As2_lgbm(y_true, y_predt) -> Tuple[np.ndarray, np.ndarray]:
    def As2_loss(x, label):
        p = sigmoid(x)
        p = np.clip(p, 10e-7, 1-10e-7)
        return -( label*(p - 1.) + (1. - label)*(p + np.log(1. - p)) )

    loss = lambda x: As2_loss(x, y_true)
    gradient = derivative(loss, y_predt, n = 1, dx = 1e-6)
    hessian = derivative(loss, y_predt, n = 2, dx = 1e-6)
    return gradient, hessian

# Define the As2 evaluation metric
def As2_err_lgbm(y_true, y_hat):
    """ Compute the evaluation function associated with the As2 loss function
    """
    prob = sigmoid(y_hat)
    prob = np.clip(prob,10e-7, 1-10e-7)
    score_fn = y_true * (prob - 1.)
    score_fp = (1. - y_true) * ( prob + np.log(1. - prob) )
    return  'As2_err_lgbm', np.mean(-(score_fn + score_fp)), False

# ------------------------------------------------------------------------------- Use the boosting loss function of  Buja et al. (2005) --------------------------------------------------------------------------------------------------- #
def update_Boost_lgbm(y_true, y_predt) -> Tuple[np.ndarray, np.ndarray]:
    def Boosting_loss(x, label):
        p = sigmoid(x)
        return label * np.sqrt( (1. - p) / (p + 1e-5) ) + (1 - label) * np.sqrt( p / (1. - p + 1e-5) )

    loss = lambda x: Boosting_loss(x, y_true)
    gradient = derivative(loss, y_predt, n = 1, dx = 1e-6)
    hessian = derivative(loss, y_predt, n = 2, dx = 1e-6)
    return gradient, hessian

# Define the Boosting evaluation metric
def Boost_err_lgbm(y_true, y_hat):
    """ Compute the evaluation function associated with the Boosting loss function
    """
    prob = sigmoid(y_hat)
    prob = np.clip(prob,10e-7, 1-10e-7)
    score_fn = y_true * np.sqrt( (1. - prob) / (prob + 1e-5) )
    score_fp = (1. - y_true) * np.sqrt( prob / (1. - prob + 1e-5) )
    return  'Boost_err_lgbm', np.mean(score_fn + score_fp), False

# ---------------------------------------------------------------------------------------------------- Use the Brier loss function ------------------------------------------------------------------------------------------------------------ #
def update_Brier_lgbm(y_true, y_predt) -> Tuple[np.ndarray, np.ndarray]:
    def Brier_loss(x, label):
        p = sigmoid(x)
        return label * np.power(1. - p, 2.) + (1 - label) * np.power(p, 2.)

    loss = lambda x: Brier_loss(x, y_true)
    gradient = derivative(loss, y_predt, n = 1, dx = 1e-6)
    hessian = derivative(loss, y_predt, n = 2, dx = 1e-6)
    return gradient, hessian

# Define the Brier evaluation metric
def Brier_err_lgbm(y_true, y_hat):
    """ Compute the evaluation function associated with the Brier loss function
    """
    prob = sigmoid(y_hat)
    prob = np.clip(prob,10e-7, 1-10e-7)
    score_fn = y_true * np.power(1. - prob, 2.)
    score_fp = (1. - y_true) * np.power(prob, 2.)
    
    # eval_name, eval_result, is_higher_better
    return  'Brier_err_lgbm', np.mean(score_fn + score_fp), False

# ---------------------------------------------------------------------------------------- Use various score functions to be used with custom loss functions  ---------------------------------------------------------------------- #
# Define the custom negative cross-entropy score function
def neg_log_loss_score_lgbm(model, X, y) -> np.float32:
    """ Compute the negative cross-entropy score function
    """
    p = sigmoid( model.predict(X) ) 
    p = np.clip(p, 10e-7, 1-10e-7)
    score_fn = y * np.log(p)
    score_fp = (1. - y) * np.log(1. - p)
    return np.mean(score_fn + score_fp)

# Define the custom As1 score function
def As1_score_lgbm(model, X, y) -> np.float32:
    """ Compute the As1 score function
    """
    p = sigmoid( model.predict(X) ) 
    p = np.clip(p, 10e-7, 1-10e-7)
    score_fn = y * (np.log(p) - p + 1)
    score_fp = (1. - y) * p
    return np.mean(score_fn - score_fp)

# Define the custom As2 score function
def As2_score_lgbm(model, X, y) -> np.float32:
    """ Compute the As2 score function
    """
    p = sigmoid( model.predict(X) ) 
    p = np.clip(p, 10e-7, 1-10e-7)
    score_fn = y * (p - 1.)
    score_fp = (1. - y) * ( p + np.log(1-p) )
    return np.mean(score_fn + score_fp)

# Define the custom boosting score function
def Boost_score_lgbm(model, X, y) -> np.float32:
    """ Compute the boosting score function
    """
    p = sigmoid( model.predict(X) ) 
    p = np.clip(p, 10e-7, 1-10e-7)
    score_fn = -y * np.sqrt( (1. - p) / (p + 1e-5) )
    score_fp = -(1. - y) * np.sqrt( p / (1. - p + 1e-5) )
    return np.mean(score_fn + score_fp)

# Define the Brier score function
def Brier_score_lgbm(model, X, y) -> np.float32:
    """ Compute the Brier score function
    """
    p = sigmoid( model.predict(X) ) 
    p = np.clip(p, 10e-7, 1-10e-7)
    score_fn = -y * np.power(1. - p, 2.)
    score_fp = -(1 - y) * np.power(p, 2.)
    return np.mean(score_fn + score_fp)

# Define the binary logloss score function to be used with a custom objective function
def accuracy_score_lgbm(model, X, y) -> np.float32:
    """ Compute the accuracy score
    """
    p = sigmoid( model.predict(X) )
    p = np.clip(p, 10e-7, 1-10e-7)
    y_pred_binary = (p > 0.5).astype('int')
    return accuracy_score(y, y_pred_binary)

# Define the AUC score
def roc_auc_lgbm(model, X, y) -> np.float32:
    """ Compute the AUC score
    """
    p = sigmoid( model.predict(X) )
    score = roc_auc_score(y, p)
    # print('auc score = ', score)
    return score

# Define the precision score
def precision_score_lgbm(model, X, y) -> np.float32:
    """ Compute the precision score
    """
    p = sigmoid( model.predict(X) )
    p = np.clip(p, 10e-7, 1-10e-7)
    y_pred_binary = (p > 0.5).astype('int')
    return precision_score(y, y_pred_binary, average = 'micro')

# Define the average precision score
def average_precision_score_lgbm(model, X, y) -> np.float32:
    """ Compute the average precision score
    """
    p = sigmoid( model.predict(X) )
    p = np.clip(p, 10e-7, 1-10e-7)
    y_pred_binary = (p > 0.5).astype('int')
    return average_precision_score(y, y_pred_binary, average = 'micro')

# Define the F1 score
def f1_score_lgbm(model, X, y) -> np.float32:
    """ Compute the F1 score
    """
    p = sigmoid( model.predict(X) )
    p = np.clip(p, 10e-7, 1-10e-7)
    y_pred_binary = (p > 0.5).astype('int')
    return f1_score(y, y_pred_binary, average = 'micro')

# Define the gain-to-pain ratio score
def gain_to_pain_ratio_score_lgbm(model, 
                                                            X, 
                                                            y, 
                                                            data_df, 
                                                            init_wealth = 1000, 
                                                            use_fixed_trans_cost = True, 
                                                            fixed_trans_cost = 10,
                                                            variable_trans_cost = 0.005,
                                                            ) -> np.float32:
    ''' Compute the gain-to-pain ratio score of a trading strategy with fixed/variable transaction cost
    INPUT
        X: a numpy array of features
        y: a pandas series of labels
        data_df: a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags ('tau' is the forecast horizon)
        init_wealth: an initial wealth
        fixed_trans_cost: the dollar amount of fixed transaction cost 
        variable_trans_cost: an amount of transaction cost as the percentage of stock price
    OUTPUT
        a float number value of the gain-to-pain ratio
    '''
    assert(variable_trans_cost >= 0 and variable_trans_cost <= 1.), 'the variable transaction cost must be greater than zero and less than one!'

    # calculate probability forecasts
    p = sigmoid( model.predict(X) )
    p = np.clip(p, 10e-7, 1-10e-7)

    # create a dataframe used to calculate trading profit/losses
    df = pd.DataFrame({'date': y.index})
    df = pd.merge(df, data_df, how = 'left', on = 'date')
    df['proba_forecast'] = p
    df.dropna(inplace = True)
    # display(df)

    if use_fixed_trans_cost:
        # implement a trading strategy with fixed transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_fixed_trans_cost(df, W0 = init_wealth, trans_cost = fixed_trans_cost, trans_date_column = 'date')
    else:
        # implement a trading strategy with variable transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_variable_trans_cost( df, W0 = init_wealth, trans_cost = variable_trans_cost, trans_date_column = 'date')

    ratio = pf.gain_to_pain_ratio( perform_df.dropna() )
    # print('ratio = ', ratio)

    del df
    gc.collect()

    return ratio


# Define the Calmar ratio
def calmar_ratio_score_lgbm(model,
                                                X, 
                                                y, 
                                                data_df,
                                                init_wealth = 1000, 
                                                use_fixed_trans_cost = True, 
                                                fixed_trans_cost = 10,
                                                variable_trans_cost = 0.005,
                                                ) -> np.float32:
    ''' Compute the Calmar ratio score of a trading strategy with fixed transaction cost
    INPUT
        X: a numpy array of features
        y: a pandas series of labels
        data_df: a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags ('tau' is the forecast horizon)
        init_wealth: an initial wealth
        fixed_trans_cost: the dollar amount of fixed transaction cost 
        variable_trans_cost: an amount of transaction cost as the percentage of stock price
    OUTPUT
        a float number value of the Calmar ratio
    '''
    assert(variable_trans_cost >= 0 and variable_trans_cost <= 1.), 'the variable transaction cost must be greater than zero and less than one!'

    # calculate probability forecasts
    p = sigmoid( model.predict(X) )
    p = np.clip(p, 10e-7, 1-10e-7)

    # create a dataframe used to calculate trading profit/losses
    df = pd.DataFrame({'date': y.index})
    df = pd.merge(df, data_df, how = 'left', on = 'date')
    df['proba_forecast'] = p
    df.dropna(inplace = True)
    # display(df)

    if use_fixed_trans_cost:
        # implement a trading strategy with fixed transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_fixed_trans_cost(df, W0 = init_wealth, trans_cost = fixed_trans_cost, trans_date_column = 'date')
    else:
        # implement a trading strategy with variable transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_variable_trans_cost( df, W0 = init_wealth, trans_cost = variable_trans_cost, trans_date_column = 'date')

    ratio = pf.calmar_ratio(perform_df)
    # print('ratio = ', ratio)

    del df
    gc.collect()

    return ratio

# Define the Sharpe ratio
def sharpe_ratio_score_lgbm(model, 
                                                X, 
                                                y, 
                                                data_df,
                                                init_wealth = 1000, 
                                                use_fixed_trans_cost = True, 
                                                fixed_trans_cost = 10,
                                                variable_trans_cost = 0.005,
                                                ) -> np.float32:
    ''' Compute the Sharpe ratio score of a trading strategy with fixed/variable transaction cost
    INPUT
        X: a numpy array of features
        y: a pandas series of labels
        data_df: a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags ('tau' is the forecast horizon)
        init_wealth: an initial wealth
        fixed_trans_cost: the dollar amount of fixed transaction cost 
        variable_trans_cost: an amount of transaction cost as the percentage of stock price
    OUTPUT
        a float number value of the Sharpe ratio
    '''
    assert(variable_trans_cost >= 0 and variable_trans_cost <= 1.), 'the variable transaction cost must be greater than zero and less than one!'

    # calculate probability forecasts
    p = sigmoid( model.predict(X) )
    p = np.clip(p, 10e-7, 1-10e-7)

    # create a dataframe used to calculate trading profit/losses
    df = pd.DataFrame({'date': y.index})
    df = pd.merge(df, data_df, how = 'left', on = 'date')
    df['proba_forecast'] = p
    df.dropna(inplace = True)
    # display(df)

    if use_fixed_trans_cost:
        # implement a trading strategy with fixed transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_fixed_trans_cost(df, W0 = init_wealth, trans_cost = fixed_trans_cost, trans_date_column = 'date')
    else:
        # implement a trading strategy with variable transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_variable_trans_cost( df, W0 = init_wealth, trans_cost = variable_trans_cost, trans_date_column = 'date')

    ratio = pf.sharpe_ratio(perform_df)
    # print('ratio = ', ratio)

    del df
    gc.collect()

    return ratio

# Define the Sortino ratio
def sortino_ratio_score_lgbm(model, 
                                                X, 
                                                y, 
                                                data_df,
                                                init_wealth = 1000, 
                                                use_fixed_trans_cost = True, 
                                                fixed_trans_cost = 10,
                                                variable_trans_cost = 0.005,
                                                ) -> np.float32:
    ''' Compute the Sortino ratio score of a trading strategy with fixed/variable transaction cost
    INPUT
        X: a numpy array of features
        y: a pandas series of labels
        data_df: a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags ('tau' is the forecast horizon)
        init_wealth: an initial wealth
        fixed_trans_cost: the dollar amount of fixed transaction cost 
        variable_trans_cost: an amount of transaction cost as the percentage of stock price
    OUTPUT
        a float number value of the Sortino ratio
    '''
    assert(variable_trans_cost >= 0 and variable_trans_cost <= 1.), 'the variable transaction cost must be greater than zero and less than one!'

    # calculate probability forecasts
    p = sigmoid( model.predict(X) )
    p = np.clip(p, 10e-7, 1-10e-7)

    # create a dataframe used to calculate trading profit/losses
    df = pd.DataFrame({'date': y.index})
    df = pd.merge(df, data_df, how = 'left', on = 'date')
    df['proba_forecast'] = p
    df.dropna(inplace = True)
    # display(df)

    if use_fixed_trans_cost:
        # implement a trading strategy with fixed transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_fixed_trans_cost(df, W0 = init_wealth, trans_cost = fixed_trans_cost, trans_date_column = 'date')
    else:
        # implement a trading strategy with variable transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_variable_trans_cost( df, W0 = init_wealth, trans_cost = variable_trans_cost, trans_date_column = 'date')

    ratio = pf.sortino_ratio(perform_df)
    # print('ratio = ', ratio)

    del df
    gc.collect()

    return ratio

# Define the correlation between equity curve and perfect profit (CECPP)
def cecpp_lgbm(model, 
                            X, 
                            y, 
                            data_df,
                            init_wealth = 1000, 
                            use_fixed_trans_cost = True, 
                            fixed_trans_cost = 10,
                            variable_trans_cost = 0.005,
                            ) -> np.float32:
    ''' Compute the correlation between equity curve and perfect profit (CECPP)
    INPUT
        X: a numpy array of features
        y: a pandas series of labels
        data_df: a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags ('tau' is the forecast horizon)
        init_wealth: an initial wealth
        fixed_trans_cost: the dollar amount of fixed transaction cost 
        variable_trans_cost: an amount of transaction cost as the percentage of stock price
    OUTPUT
        a float number value of the CECPP
    '''
    assert(variable_trans_cost >= 0 and variable_trans_cost <= 1.), 'the variable transaction cost must be greater than zero and less than one!'

    # calculate probability forecasts
    p = sigmoid( model.predict(X) )
    p = np.clip(p, 10e-7, 1-10e-7)

    # create a dataframe used to calculate trading profit/losses
    df = pd.DataFrame({'date': y.index})
    df = pd.merge(df, data_df, how = 'left', on = 'date')
    df['proba_forecast'] = p
    df.dropna(inplace = True)
    # display(df)

    if use_fixed_trans_cost:
        # implement a trading strategy with fixed transaction cost
        wealth_bh1, ret_bh1, perform_df1 = tt.perfect_profit_fixed_trans_cost(df, W0 = init_wealth, trans_cost = fixed_trans_cost, trans_date_column = 'date')
        wealth_bh2, ret_bh2, perform_df2 = tt.trade_fixed_trans_cost(df, W0 = init_wealth, trans_cost = fixed_trans_cost, trans_date_column = 'date')
    else:
        # implement a trading strategy with variable transaction cost
        wealth_bh1, ret_bh1, perform_df1 = tt.perfect_profit_variable_trans_cost(df, W0 = init_wealth, trans_cost = variable_trans_cost, trans_date_column = 'date')
        wealth_bh2, ret_bh2, perform_df2 = tt.trade_variable_trans_cost(df, W0 = init_wealth, trans_cost = variable_trans_cost, trans_date_column = 'date')

    cecpp_value = pf.cecpp( perform_df1.dropna(), perform_df2.dropna() )
    # print('CECPP = ', cecpp_value)
    
    del df
    gc.collect()

    return cecpp_value


# ============================================ Finish defining the custom loss functions and evaluation metrics for LightGBM ========================================= #

# ======================================== Define score functions to be used with the built-in loss functions in the Scikit-learn API ======================================== #
# Define the As1 score function to use with sklearn
def As1_score_sklearn(y_true, y_prob):
    """ Compute the As1 score function
    """
    # print('y_prob = ', y_prob)
    # print('y_true = ', y_true)
    p = np.clip(y_prob, 10e-7, 1-10e-7)
    score_fn = y_true * (np.log(p) - p + 1)
    score_fp = (1. - y_true) * p
    score = np.mean(score_fn - score_fp)
    # print('As1 score = ', score)
    return score
As1_score_sklearn = make_scorer(As1_score_sklearn, greater_is_better = True, needs_proba=True)

# Define the As2 score function to use with sklearn
def As2_score_sklearn(y_true, y_prob):
    """ Compute the As2 score function
    """
    # print('y_prob = ', y_prob)
    # print('y_true = ', y_true)
    p = np.clip(y_prob, 10e-7, 1-10e-7)
    score_fn = y_true * (p - 1.)
    score_fp = (1. - y_true) * (p + np.log(1. - p))
    return np.mean(score_fn + score_fp)
As2_score_sklearn = make_scorer(As2_score_sklearn, greater_is_better = True, needs_proba=True)

# Define the boosting score function
def Boost_score_sklearn(y_true, y_prob):
    """ Compute the boost score of Buja et al. (2005)
    """
    p = np.clip(y_prob, 10e-7, 1-10e-7)
    score_fn = -y_true * np.sqrt( (1. - p) / p )
    score_fp = -(1. - y_true) * np.sqrt( p / (1. - p) )
    return np.mean(score_fn + score_fp)
Boost_score_sklearn = make_scorer(Boost_score_sklearn, greater_is_better = True, needs_proba=True)

# Define the Brier score function
def Brier_score_sklearn(y_true, y_prob):
    """ Compute the Brier score function
    """
    p = np.clip(y_prob, 10e-7, 1-10e-7)
    score_fn = -y_true * np.power(1. - p, 2.)
    score_fp = -(1. - y_true) * np.power(p, 2.)
    return np.mean(score_fn + score_fp)
Brier_score_sklearn = make_scorer(Brier_score_sklearn, greater_is_better = True, needs_proba=True)

# Define the gain-to-pain ratio score
def gain_to_pain_ratio_score_sklearn(y_true, 
                                                                y_prob,
                                                                data_df,
                                                                init_wealth = 1000, 
                                                                use_fixed_trans_cost = True, 
                                                                fixed_trans_cost = 10,
                                                                variable_trans_cost = 0.005,
                                                                ) -> np.float32:
    ''' Compute the gain-to-pain ratio score of a trading strategy with fixed/variable transaction cost
    INPUT
        y_true: a pandas series of labels
        y_prob: a numpy array of probability forecasts
        data_df: a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags ('tau' is the forecast horizon)
        init_wealth: an initial wealth
        fixed_trans_cost: the dollar amount of fixed transaction cost 
        variable_trans_cost: an amount of transaction cost as the percentage of stock price
    OUTPUT
        a float number value of the gain-to-pain ratio
    '''
    assert(variable_trans_cost >= 0 and variable_trans_cost <= 1.), 'the variable transaction cost must be greater than zero and less than one!'

    # create a dataframe used to calculate trading profit/losses
    df = pd.DataFrame({'date': y_true.index})
    df = pd.merge(df, data_df, how = 'left', on = 'date')
    df['proba_forecast'] = np.clip(y_prob, 10e-7, 1-10e-7)
    df.dropna(inplace = True)
    # display(df)

    if use_fixed_trans_cost:
        # implement a trading strategy with fixed transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_fixed_trans_cost(df, W0 = init_wealth, trans_cost = fixed_trans_cost, trans_date_column = 'date')
    else:
        # implement a trading strategy with variable transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_variable_trans_cost( df, W0 = init_wealth, trans_cost = variable_trans_cost, trans_date_column = 'date')

    ratio = pf.gain_to_pain_ratio(perform_df)
    # print('ratio = ', ratio)

    del df
    gc.collect()

    return ratio


# Define the Calmar ratio
def calmar_ratio_score_sklearn(y_true, 
                                                    y_prob,
                                                    data_df,
                                                    init_wealth = 1000, 
                                                    use_fixed_trans_cost = True, 
                                                    fixed_trans_cost = 10,
                                                    variable_trans_cost = 0.005,
                                                    ) -> np.float32:
    ''' Compute the Calmar ratio score of a trading strategy with fixed/variable transaction cost
    INPUT
        y_true: a pandas series of labels
        y_prob: a numpy array of probability forecasts
        data_df: a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags ('tau' is the forecast horizon)
        init_wealth: an initial wealth
        fixed_trans_cost: the dollar amount of fixed transaction cost 
        variable_trans_cost: an amount of transaction cost as the percentage of stock price
    OUTPUT
        a float number value of the Calmar ratio
    '''
    assert(variable_trans_cost >= 0 and variable_trans_cost <= 1.), 'the variable transaction cost must be greater than zero and less than one!'

    # create a dataframe used to calculate trading profit/losses
    df = pd.DataFrame({'date': y_true.index})
    df = pd.merge(df, data_df, how = 'left', on = 'date')
    df['proba_forecast'] = np.clip(y_prob, 10e-7, 1-10e-7)
    df.dropna(inplace = True)
    # display(df)

    if use_fixed_trans_cost:
        # implement a trading strategy with fixed transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_fixed_trans_cost(df, W0 = init_wealth, trans_cost = fixed_trans_cost, trans_date_column = 'date')
    else:
        # implement a trading strategy with variable transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_variable_trans_cost( df, W0 = init_wealth, trans_cost = variable_trans_cost, trans_date_column = 'date')

    ratio = pf.calmar_ratio(perform_df)
    # print('ratio = ', ratio)

    del df
    gc.collect()

    return ratio


# Define the Sharpe ratio
def sharpe_ratio_score_sklearn( y_true, 
                                                    y_prob,
                                                    data_df,
                                                    init_wealth = 1000, 
                                                    use_fixed_trans_cost = True, 
                                                    fixed_trans_cost = 10,
                                                    variable_trans_cost = 0.005,
                                                    ) -> np.float32:
    ''' Compute the Sharpe ratio score of a trading strategy with fixed/variable transaction cost
    INPUT
        y_true: a pandas series of labels
        y_prob: a numpy array of probability forecasts
        data_df: a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags ('tau' is the forecast horizon)
        init_wealth: an initial wealth
        fixed_trans_cost: the dollar amount of fixed transaction cost 
        variable_trans_cost: an amount of transaction cost as the percentage of stock price
    OUTPUT
        a float number value of the Sharpe ratio
    '''
    assert(variable_trans_cost >= 0 and variable_trans_cost <= 1.), 'the variable transaction cost must be greater than zero and less than one!'

    # create a dataframe used to calculate trading profit/losses
    df = pd.DataFrame({'date': y_true.index})
    df = pd.merge(df, data_df, how = 'left', on = 'date')
    df['proba_forecast'] = np.clip(y_prob, 10e-7, 1-10e-7)
    df.dropna(inplace = True)
    # display(df)

    if use_fixed_trans_cost:
        # implement a trading strategy with fixed transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_fixed_trans_cost(df, W0 = init_wealth, trans_cost = fixed_trans_cost, trans_date_column = 'date')
    else:
        # implement a trading strategy with variable transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_variable_trans_cost( df, W0 = init_wealth, trans_cost = variable_trans_cost, trans_date_column = 'date')

    ratio = pf.sharpe_ratio(perform_df)
    # print('ratio = ', ratio)

    del df
    gc.collect()

    return ratio

# Define the Sortino ratio
def sortino_ratio_score_sklearn( y_true, 
                                                    y_prob,
                                                    data_df,
                                                    init_wealth = 1000, 
                                                    use_fixed_trans_cost = True, 
                                                    fixed_trans_cost = 10,
                                                    variable_trans_cost = 0.005,
                                                    ) -> np.float32:
    ''' Compute the Sortino ratio score of a trading strategy with fixed/variable transaction cost
    INPUT
        y_true: a pandas series of labels
        y_prob: a numpy array of probability forecasts
        data_df: a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags ('tau' is the forecast horizon)
        init_wealth: an initial wealth
        fixed_trans_cost: the dollar amount of fixed transaction cost 
        variable_trans_cost: an amount of transaction cost as the percentage of stock price
    OUTPUT
        a float number value of the Sharpe ratio
    '''
    assert(variable_trans_cost >= 0 and variable_trans_cost <= 1.), 'the variable transaction cost must be greater than zero and less than one!'

    # create a dataframe used to calculate trading profit/losses
    df = pd.DataFrame({'date': y_true.index})
    df = pd.merge(df, data_df, how = 'left', on = 'date')
    df['proba_forecast'] = np.clip(y_prob, 10e-7, 1-10e-7)
    df.dropna(inplace = True)
    # display(df)

    if use_fixed_trans_cost:
        # implement a trading strategy with fixed transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_fixed_trans_cost(df, W0 = init_wealth, trans_cost = fixed_trans_cost, trans_date_column = 'date')
    else:
        # implement a trading strategy with variable transaction cost
        wealth_bh, ret_bh, perform_df = tt.trade_variable_trans_cost( df, W0 = init_wealth, trans_cost = variable_trans_cost, trans_date_column = 'date')

    ratio = pf.sortino_ratio(perform_df)
    # print('ratio = ', ratio)

    del df
    gc.collect()

    return ratio

# Define the correlation between equity curve and perfect profit (CECPP)
def cecpp_sklearn( y_true, 
                                y_prob,
                                data_df,
                                init_wealth = 1000, 
                                use_fixed_trans_cost = True, 
                                fixed_trans_cost = 10,
                                variable_trans_cost = 0.005,
                                ) -> np.float32:
    ''' Compute the correlation between equity curve and perfect profit (CECPP) of a trading strategy with fixed/variable transaction cost
    INPUT
        y_true: a pandas series of labels
        y_prob: a numpy array of probability forecasts
        data_df: a pandas dataframe consisting of columns: 'date', 'price', and 'RF' with the last two columns being shifted forward 'tau' lags ('tau' is the forecast horizon)
        init_wealth: an initial wealth
        fixed_trans_cost: the dollar amount of fixed transaction cost 
        variable_trans_cost: an amount of transaction cost as the percentage of stock price
    OUTPUT
        a float number value of the Calmar ratio
    '''
    assert(variable_trans_cost >= 0 and variable_trans_cost <= 1.), 'the variable transaction cost must be greater than zero and less than one!'

    # create a dataframe used to calculate trading profit/losses
    df = pd.DataFrame({'date': y_true.index})
    df = pd.merge(df, data_df, how = 'left', on = 'date')
    df['proba_forecast'] = np.clip(y_prob, 10e-7, 1-10e-7)
    df.dropna(inplace = True)
    # display(df)

    if use_fixed_trans_cost:
        # implement a trading strategy with fixed transaction cost
        wealth_bh1, ret_bh1, perform_df1 = tt.perfect_profit_fixed_trans_cost(df, W0 = init_wealth, trans_cost = fixed_trans_cost, trans_date_column = 'date')
        wealth_bh2, ret_bh2, perform_df2 = tt.trade_fixed_trans_cost(df, W0 = init_wealth, trans_cost = fixed_trans_cost, trans_date_column = 'date')
    else:
        # implement a trading strategy with variable transaction cost
        wealth_bh1, ret_bh1, perform_df1 = tt.perfect_profit_variable_trans_cost(df, W0 = init_wealth, trans_cost = variable_trans_cost, trans_date_column = 'date')
        wealth_bh2, ret_bh2, perform_df2 = tt.trade_variable_trans_cost(df, W0 = init_wealth, trans_cost = variable_trans_cost, trans_date_column = 'date')

    cecpp_value = pf.cecpp( perform_df1.dropna(), perform_df2.dropna() )
    # print('cecpp = ', cecpp_value)

    del df
    gc.collect()

    return cecpp_value
# ======================================== Finish defining score functions to be used with the built-in loss functions in the Scikit-learn API ======================================== #
