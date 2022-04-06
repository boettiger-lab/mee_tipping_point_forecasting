import sys
sys.path.append("../")
from model import stochastic_tp, saddle_node_tp
import numpy as np
from darts import TimeSeries
from pandas import RangeIndex

models = {"stochastic" : stochastic_tp, 
          "saddle" : saddle_node_tp,
          }

# Need to change this so that it doesn't intake args
def preprocessed_t_series(model, n_samples, random_alpha=False):
    _data = models[model.lower()]()
    
    # Generating time series
    if random_alpha:
        training_data = _data.collect_samples(n_samples, random_alpha)
    else:
        training_data = _data.collect_samples(n_samples)

    # Preprocessing time series to
    _ts = [[] for i in range(training_data.shape[1])]
    for i in range(n_samples):
        for j in range(training_data.shape[1]):
            _ts[j].append(training_data[i, j])

    vals = np.array(_ts).reshape(training_data.shape[1], 1, n_samples)
    return TimeSeries.from_times_and_values(RangeIndex(250),vals), vals

def truth_dist(model, t_series, input_len, output_len, n_samples=100, random_alpha=False):
    N_init = t_series[-1].values()[0][0]
    t_max = 250-input_len
        
    _data = models[model.lower()](N_init, t_max)
    
    # Need to account for degradation parameter having changed over t_series
    # for saddle bifurcations
    if model == "saddle":
        _data.h = _data.h_init + len(t_series) * _data.alpha
        _data.h_init = _data.h_init + len(t_series) * _data.alpha

    for i in range(n_samples):
      if random_alpha:
          training_data = _data.collect_samples(n_samples, random_alpha)
      else:
          training_data = _data.collect_samples(n_samples)
      
      # Preprocessing time series to
      _ts = [[] for i in range(training_data.shape[1])]
      for i in range(n_samples):
          for j in range(training_data.shape[1]):
              _ts[j].append(training_data[i, j])
  
      vals = np.array(_ts).reshape(training_data.shape[1], 1, n_samples)
    return TimeSeries.from_times_and_values(RangeIndex(start=input_len, stop=250), vals)
        
