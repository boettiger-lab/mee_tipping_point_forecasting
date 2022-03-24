import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import tipping_point
from darts import TimeSeries
from darts.models import TransformerModel
from darts.utils.likelihood_models import LaplaceLikelihood
from utils import preprocessed_t_series

# Generating time series
train_series = preprocessed_t_series(100)

my_model = TransformerModel(
    batch_size=32,
    n_epochs=100,
    likelihood=LaplaceLikelihood(),
    optimizer_kwargs={"lr": 1e-2},
    model_name="tipping_transformer_test",
    log_tensorboard=True,
    d_model=16,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    activation="relu",
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

# Generating time series
for i in range(5):
    train_series = preprocessed_t_series(1)
    train_series.plot(label="training")
    pred = my_model.historical_forecasts(train_series, retrain=False, start=25, num_samples=50)
    pred.plot(low_quantile=0.01, high_quantile=0.99, label="1-99th percentiles")
    plt.savefig(f"plots/transformer_test_{i}")
    plt.clf()
