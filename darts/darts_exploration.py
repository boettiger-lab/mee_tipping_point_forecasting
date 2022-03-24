import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import tipping_point
from darts import TimeSeries
from darts.models import RNNModel, TCNModel
from darts.utils.likelihood_models import LaplaceLikelihood
from utils import preprocessed_t_series

# Generating time series
train_series = preprocessed_t_series(100)

my_model = TCNModel(
    batch_size=8,
    dropout=0,
    n_epochs=200,
    dilation_base=4,
    num_layers=3,
    kernel_size=4,
    likelihood=LaplaceLikelihood(),
    optimizer_kwargs={"lr": 1e-2},
    model_name="best_tuned_tcn",
    log_tensorboard=True,
    random_state=42,
    input_chunk_length=25,
    output_chunk_length=24,
    force_reset=True,
    save_checkpoints=True,
    torch_device_str = "cuda:0",
)


my_model.fit(
    train_series,
    verbose=True,
)

# Generating time series
for i in range(5):
    train_series = preprocessed_t_series(1)
    train_series.plot(label="training")
    pred = my_model.historical_forecasts(train_series, retrain=False, start=25, num_samples=50)
    pred.plot(low_quantile=0.01, high_quantile=0.99, label="1-99th percentiles")
    plt.savefig(f"plots/best_tuned_tcn_200_epochs_{i}")
    plt.clf()
