import pandas as pd
import numpy as np
import sys
sys.path.append("../")
from model import tipping_point
from darts import TimeSeries
from darts.models import RNNModel

# Loading data which is not in a convenient format so I need to do some processing
_data = tipping_point()
n_samples = 1
training_data = _data.collect_samples(n_samples)

_ts = [[] for i in range(training_data.shape[1])]
for i in range(n_samples):
    for j in range(training_data.shape[1]):
        _ts[j].append(training_data[i, j])

ts = np.array(_ts).reshape(training_data.shape[1], 1, n_samples)
train_series = TimeSeries.from_values(ts)
train_series

my_model = RNNModel(
      model="LSTM",
      input_chunk_length=25,
)


my_model.load_model("darts_logs/tipping_lstm_test/_model.pth.tar")
my_model.fit(train_series)
pred = my_model.predict(n=75, num_samples=500)
pred.plot(low_quantile=0.2, high_quantile=0.8, label="20-80th percentiles")
import pdb; pdb.set_trace()
