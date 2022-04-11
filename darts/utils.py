import sys
sys.path.append("../")
from model import stochastic_tp, saddle_node_tp
import numpy as np
from darts import TimeSeries
from pandas import RangeIndex, DataFrame

models = {"stochastic" : stochastic_tp, 
          "saddle" : saddle_node_tp,
          }

# Need to change this so that it doesn't intake args
def preprocessed_t_series(model, n_samples):
    _data = models[model.lower()]()
    training_data = _data.collect_samples(n_samples)

    # Preprocessing time series to
    _ts = [[] for i in range(training_data.shape[1])]
    for i in range(n_samples):
        for j in range(training_data.shape[1]):
            _ts[j].append(training_data[i, j])

    vals = np.array(_ts).reshape(training_data.shape[1], 1, n_samples)
    return TimeSeries.from_times_and_values(RangeIndex(250),vals), vals.reshape(250, n_samples).T

def truth_dist(model, t_series, input_len, output_len, n_samples=100):
    N_init = t_series[-1].values()[0][0]
    t_max = 250-input_len
        
    _data = models[model.lower()](N_init, t_max)
    
    # Need to account for degradation parameter having changed over t_series
    # for saddle bifurcations
    if model == "saddle":
        _data.h = _data.h_init + len(t_series) * _data.alpha
        _data.h_init = _data.h_init + len(t_series) * _data.alpha

    for i in range(n_samples):
      training_data = _data.collect_samples(n_samples)
      
      # Preprocessing time series to
      _ts = [[] for i in range(training_data.shape[1])]
      for i in range(n_samples):
          for j in range(training_data.shape[1]):
              _ts[j].append(training_data[i, j])
  
      vals = np.array(_ts).reshape(training_data.shape[1], 1, n_samples)
    return TimeSeries.from_times_and_values(RangeIndex(start=input_len, stop=250), vals)
  
def count_tipped(vals):
    n_vals = len(vals)
    count = 0
    for i in range(n_vals):
        if vals[i, -1] < 0.6:
            count += 1
    return count / n_vals

def import_hypers(hyper_file_name):
    if hyper_file_name == "lstm":
        from train_hyperparams.lstm import hyperparameters
    elif hyper_file_name == "tcn":
        from train_hyperparams.tcn import hyperparameters
    elif hyper_file_name == "transformer":
        from train_hyperparams.transformer import hyperparameters
    elif hyper_file_name == "stochastic_10":
        from train_hyperparams.stochastic_10 import hyperparameters
    elif hyper_file_name == "stochastic_100":
        from train_hyperparams.stochastic_100 import hyperparameters
    elif hyper_file_name == "stochastic_1000":
        from train_hyperparams.stochastic_1000 import hyperparameters
    elif hyper_file_name == "saddle_10":
        from train_hyperparams.saddle_10 import hyperparameters
    elif hyper_file_name == "saddle_100":
        from train_hyperparams.saddle_100 import hyperparameters
    elif hyper_file_name == "saddle_1000":
        from train_hyperparams.saddle_1000 import hyperparameters

def make_df(prediction_series, truth_series, tp_model, ml_model, case, n_samples):
    predictions_array = prediction_series.all_values()[:, 0, :]
    truth_array = truth_series.all_values()[:, 0, :]
    
    predictions_df = DataFrame(predictions_array)
    truth_df = DataFrame(truth_array)
    
    predictions_df["time"] = prediction_series.time_index
    truth_df["time"] = truth_series.time_index
    
    predictions_df = predictions_df.melt(id_vars="time", var_name="ensemble", value_name="value")
    truth_df = truth_df.melt(id_vars="time", var_name="ensemble", value_name="value")
    
    
    predictions_df["tp_model"] = tp_model
    predictions_df["ml_model"] = ml_model
    predictions_df["case"] = n_samples if case == "none" else case
    predictions_df["true_model"] = False
    truth_df["tp_model"] = tp_model
    truth_df["ml_model"] = ml_model
    truth_df["case"] = n_samples if case == "none" else case
    truth_df["true_model"] = True
    df = predictions_df.append(truth_df, ignore_index=True)
    
    
    return df
    
  
    
    
