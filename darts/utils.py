import sys
sys.path.append("../")
from model import stochastic_tp, saddle_node_tp, hopf_tp
import numpy as np
from darts import TimeSeries
from pandas import RangeIndex, DataFrame

models = {"stochastic" : stochastic_tp, 
          "saddle" : saddle_node_tp,
          }

def get_train_series(args):
    # Selecting case to pick training set
    if args.case.lower() == "tipped" and args.tp_model == "stochastic":
        train_series = TimeSeries.from_csv("stochastic_tipped.csv", time_col="time")
    elif args.case.lower() == "nontipped" and args.tp_model == "stochastic":
        train_series = TimeSeries.from_csv("stochastic_non_tipped.csv", time_col="time")
    elif args.case.lower() == "both" and args.tp_model == "stochastic":
        train_series = TimeSeries.from_csv("stochastic_tipped.csv", time_col="time")
        train_series_ = TimeSeries.from_csv("stochastic_non_tipped.csv", time_col="time")
        train_series = train_series.concatenate(train_series_, axis="sample")
    elif args.case.lower() == "none":
        # Generating time series
        train_series, _ = preprocessed_t_series(args.tp_model, args.n_samples, reverse=args.reverse)
    
    return train_series

def preprocessed_t_series(model, n_samples, reverse=False):
    if model == "hopf":
        if reverse:
            _data = hopf_tp(K_init=30, delta=-0.08)
        else:
            _data = hopf_tp(K_init=14, delta=0.08)
    else:
        _data = models[model.lower()]()
        
    training_data = _data.collect_samples(n_samples)

    # Preprocessing time series to
    _ts = [[] for i in range(training_data.shape[1])]
    for i in range(n_samples):
        for j in range(training_data.shape[1]):
            _ts[j].append(training_data[i, j])

    vals = np.array(_ts).reshape(training_data.shape[1], 1, n_samples)
    t_max = 250 if model != "hopf" else 100
    return TimeSeries.from_times_and_values(RangeIndex(t_max),vals), vals.reshape(t_max, n_samples).T

def truth_dist(model, t_series, input_len, output_len, n_samples=100, reverse=False):
    N_init = t_series[-1].values()[0][0]
    t_max = 250-input_len if model != "hopf" else 200-input_len
    
    if model == "hopf":
        if reverse:
            _data = hopf_tp(N_init, t_max, K_init=30, delta=-0.08)
        else:
            _data = hopf_tp(N_init, t_max, K_init=14, delta=0.08)
    else:
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
    stop = 250 if model != "hopf" else 200
    return TimeSeries.from_times_and_values(RangeIndex(start=input_len, stop=250), vals)
  
def count_tipped(vals):
    n_vals = len(vals)
    count = 0
    for i in range(n_vals):
        if vals[i, -1] < 0.6:
            count += 1
    return count / n_vals


def import_hypers(tp_model, ml_model):
    if ml_model == "lstm": 
        if tp_model == "stochastic":
            from train_hyperparams.stochastic_lstm import hyperparameters
        elif tp_model == "saddle":
            from train_hyperparams.saddle_lstm import hyperparameters
        elif tp_model == "hopf":
            from train_hyperparams.hopf_lstm import hyperparameters
    elif ml_model == "gru":
        if tp_model == "stochastic":
            from train_hyperparams.stochastic_gru import hyperparameters
        elif tp_model == "saddle":
            from train_hyperparams.saddle_gru import hyperparameters
        elif tp_model == "hopf":
            from train_hyperparams.hopf_gru import hyperparameters
    elif ml_model == "block_rnn":
        if tp_model == "stochastic":
            from train_hyperparams.stochastic_block import hyperparameters
        elif tp_model == "saddle":
            from train_hyperparams.saddle_block import hyperparameters
        elif tp_model == "hopf":
            from train_hyperparams.hopf_block import hyperparameters
    elif ml_model == "transformer":
        if tp_model == "stochastic":
            from train_hyperparams.stochastic_transformer import hyperparameters
        elif tp_model == "saddle":
            from train_hyperparams.saddle_transformer import hyperparameters
        elif tp_model == "hopf":
            from train_hyperparams.hopf_transformer import hyperparameters

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
    
  
    
    
