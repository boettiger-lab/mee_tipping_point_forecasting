import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import tipping_point
from darts import TimeSeries
from darts.models import RNNModel
from darts.utils.likelihood_models import GaussianLikelihood

# Loading data
_data = tipping_point()
n_samples = 100
training_data = _data.collect_samples(n_samples)

_ts = [[] for i in range(training_data.shape[1])]
for i in range(n_samples):
    for j in range(training_data.shape[1]):
        _ts[j].append(training_data[i, j])

ts = np.array(_ts).reshape(training_data.shape[1], 1, n_samples)
train_series = TimeSeries.from_values(ts)

my_model = RNNModel(
    model="LSTM",
    hidden_dim=64,
    dropout=0.3,
    batch_size=1,
    n_epochs=10,
    likelihood=GaussianLikelihood(),
    optimizer_kwargs={"lr": 1e-3},
    model_name="tipping_lstm_test",
    log_tensorboard=False,
    random_state=42,
    input_chunk_length=25,
    output_chunk_length=75,
    force_reset=True,
    save_checkpoints=True,
    torch_device_str = "cuda:0",
)


my_model.fit(
    train_series,
    verbose=True,
)

train_series.plot(label="training")
pred = my_model.historical_forecasts(train_series, retrain=False, start=25, num_samples=500)
pred.plot(low_quantile=0.01, high_quantile=0.99, label="1-99th percentiles")
plt.savefig("trash")
