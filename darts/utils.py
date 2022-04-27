import sys
sys.path.append("../")
from model import stochastic_tp, saddle_node_tp, hopf_tp
import numpy as np
import pandas as pd
from darts import TimeSeries
from pandas import RangeIndex, DataFrame

models = { 
          "saddle" : saddle_node_tp,
          }

def get_train_series(args):
    # Selecting case to pick training set
    if args.case.lower() == "tipped" and args.sim_model == "stochastic":
        train_series = TimeSeries.from_csv("stochastic_tipped.csv", time_col="time")
    elif args.case.lower() == "nontipped" and args.sim_model == "stochastic":
        train_series = TimeSeries.from_csv("stochastic_non_tipped.csv", time_col="time")
    elif args.case.lower() == "both" and args.sim_model == "stochastic":
        train_series = TimeSeries.from_csv("stochastic_tipped.csv", time_col="time")
        train_series_ = TimeSeries.from_csv("stochastic_non_tipped.csv", time_col="time")
        train_series = train_series.concatenate(train_series_, axis="sample")
    elif args.case.lower() == "none":
        # Generating time series
        train_series = preprocessed_t_series(args.sim_model, args.n_samples, reverse=args.decrease)
    
    return train_series

def preprocessed_t_series(model, n_samples, reverse=False):
    if model == "hopf":
        if reverse:
            _data = hopf_tp(K_init=30, delta=-0.08)
        else:
            _data = hopf_tp(K_init=14, delta=0.08)
    elif model == "stochastic":
        _data = saddle_node_tp(N=0.55, alpha=0, h=0.26)
    else:
        _data = models[model.lower()]()
        
    training_data = _data.collect_samples(n_samples)
    # Note training_data.shape[1] is the length of the time series
    # Preprocessing time series to
    _ts = [[] for i in range(training_data.shape[1])]
    for i in range(n_samples):
        for j in range(training_data.shape[1]):
            _ts[j].append(training_data[i, j])
    
    t_max = 250 if model != "hopf" else 100
    
    if model == "hopf":
        _ts = np.array(_ts)
        _vals = []
        for t_slice in _ts:
            _vals.append(t_slice.T)
        vals = np.array(_vals)
    else:
        vals = np.array(_ts).reshape(training_data.shape[1], 1, n_samples)
        
    return TimeSeries.from_times_and_values(RangeIndex(t_max),vals)

def truth_dist(model, t_series, input_len, output_len, n_draws=100, reverse=False, start_t=25):
    N_init = t_series[-1].values()[0] 
    t_max = 250-input_len if model != "hopf" else 100
    t = len(t_series) if model!= "hopf" else 100
    
    if model == "hopf":
        if reverse:
            delta=-0.08
            K_init = 30 + delta * t 
            _data = hopf_tp(N_init, t_max, K_init=K_init, delta=delta)
        else:
            delta=0.08
            K_init = 14 + delta * t
            _data = hopf_tp(N_init, t_max, K_init=K_init, delta=delta)
    elif model == "stochastic":
        _data = saddle_node_tp(N=N_init, t_max=t_max, alpha=0, h=0.26)
    else:
        _data = models[model.lower()](N_init[0], t_max)
    
    # Need to account for degradation parameter having changed over t_series
    # for saddle bifurcations
    if model == "saddle":
        _data.h = _data.h_init + t * _data.alpha
        _data.h_init = _data.h_init + t * _data.alpha

    training_data = _data.collect_samples(n_draws)
    
    # Preprocessing time series to
    _ts = [[] for x in range(training_data.shape[1])]
    for j in range(n_draws):
        for k in range(training_data.shape[1]):
            _ts[k].append(training_data[j, k])
    
    if model == "hopf":
        _ts = np.array(_ts)
        _vals = []
        for t_slice in _ts:
            _vals.append(t_slice.T)
        vals = np.array(_vals)
    else:
        vals = np.array(_ts).reshape(training_data.shape[1], 1, n_draws)
        
    stop = 250 if model != "hopf" else 200
    import pdb; pdb.set_trace()
    return TimeSeries.from_times_and_values(RangeIndex(start=start_t, stop=stop), vals)
  
def count_tipped(vals):
    n_samples = vals.shape[2]
    count = 0
    for i in range(n_samples):
        if vals[-1, 0, i] < 0.3:
            count += 1
    return count / n_samples

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

def make_df(prediction_series, truth_series, historical_series, tp_model, ml_model, case, n_samples, index):
    if tp_model != "hopf":
        predictions_array = prediction_series.all_values()[:, 0, :]
        truth_array = truth_series.all_values()[:, 0, :] 
        hist_array = historical_series.all_values()[:, 0, :]
        
        predictions_df = DataFrame(predictions_array)
        truth_df = DataFrame(truth_array)
        hist_df = DataFrame(hist_array)
    
    else:
        predictions_array = reshape_xarray(prediction_series.all_values())
        truth_array = reshape_xarray(truth_series.all_values())
        hist_array = reshape_xarray(historical_series.all_values())
    
        predictions_df = make_hopf_df(predictions_array)
        truth_df = make_hopf_df(truth_array)
        hist_df = make_hopf_df(hist_array)
    
    predictions_df["t"] = prediction_series.time_index
    truth_df["t"] = truth_series.time_index
    hist_df["t"] = historical_series.time_index
  
    predictions_df = predictions_df.melt(id_vars="t", var_name="iter", value_name="value")
    truth_df = truth_df.melt(id_vars="t", var_name="iter", value_name="value")
    hist_df = hist_df.melt(id_vars="t", var_name="iter", value_name="value")
    
    if tp_model == "hopf":
        predictions_df = melt_hopf_df(predictions_df)
        truth_df = melt_hopf_df(truth_df)
        hist_df = melt_hopf_df(hist_df)
    else:
        predictions_df["variable"] = "X"
        truth_df["variable"] = "X"
        hist_df["variable"] = "X"
    
    predictions_df["simulation"] = tp_model
    predictions_df["forecasting_model"] = ml_model
    predictions_df["case"] = n_samples if case == "none" else case
    predictions_df["type"] = "predicted"
    truth_df["simulation"] = tp_model
    truth_df["forecasting_model"] = ml_model
    truth_df["case"] = n_samples if case == "none" else case
    truth_df["type"] = "true"
    hist_df["simulation"] = tp_model
    hist_df["forecasting_model"] = ml_model
    hist_df["case"] = n_samples if case == "none" else case
    hist_df["type"] = "historical"
    df = predictions_df.append(truth_df, ignore_index=True).append(hist_df, ignore_index=True)
    df["group"] = index
    
    return df
    
    
def reshape_xarray(_ts):
    _vals = []
    for t_slice in _ts:
        _vals.append(t_slice.T)
    vals = np.array(_vals)
    return vals
  
def make_hopf_df(array):
    df = pd.DataFrame()
    for i in range(array.shape[1]):
        _df = pd.DataFrame(array[:, i])
        _df[0] = _df[[0, 1]].values.tolist()
        del _df[1]
        _df = _df.rename(columns={0:i})
        df = pd.concat([df, _df], axis=1)

    return df
  
def melt_hopf_df(df):
    split_df = pd.DataFrame(df['value'].tolist(), columns=['H', 'P'])
    df = pd.concat([df, split_df], axis=1)
    del df["value"]
    df = pd.melt(df, id_vars=['t', 'iter'], value_vars=["H", "P"])
    
    return df
        
