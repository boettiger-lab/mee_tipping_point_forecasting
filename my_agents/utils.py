import sys
sys.path.append("../")
from model import tipping_point
import numpy as np
from darts import TimeSeries
from pandas import RangeIndex

def preprocessed_t_series(n_samples):
    # Generating time series
    _data = tipping_point()
    training_data = _data.collect_samples(n_samples)

    # Preprocessing time series to
    _ts = [[] for i in range(training_data.shape[1])]
    for i in range(n_samples):
        for j in range(training_data.shape[1]):
            _ts[j].append(training_data[i, j])

    vals = np.array(_ts).reshape(training_data.shape[1], 1, n_samples)
    return TimeSeries.from_times_and_values(RangeIndex(250),vals)

def truth_dist(t_series, n_samples=100):
    N_init = t_series[-1].values()[0][0]
    t_max = 125
    _data = tipping_point(N_init, t_max)
    for i in range(n_samples):
      training_data = _data.collect_samples(n_samples)
      
      # Preprocessing time series to
      _ts = [[] for i in range(training_data.shape[1])]
      for i in range(n_samples):
          for j in range(training_data.shape[1]):
              _ts[j].append(training_data[i, j])
  
      vals = np.array(_ts).reshape(training_data.shape[1], 1, n_samples)
    return TimeSeries.from_times_and_values(RangeIndex(start=25, stop=25+t_max),vals)
        
def saddlenode_t_series(n_samples):
    # Generating time series
    _data = tipping_point()
    training_data = _data.collect_samples(n_samples)

    # Preprocessing time series to
    _ts = [[] for i in range(training_data.shape[1])]
    for i in range(n_samples):
        for j in range(training_data.shape[1]):
            _ts[j].append(training_data[i, j])

    vals = np.array(_ts).reshape(training_data.shape[1], 1, n_samples)
    return TimeSeries.from_times_and_values(RangeIndex(250),vals)