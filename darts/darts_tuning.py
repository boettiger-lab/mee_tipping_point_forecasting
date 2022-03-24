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
from darts.utils.likelihood_models import LaplaceLikelihood
from utils import preprocessed_t_series

# Generating time series
series = preprocessed_t_series(100)

my_model = RNNModel(
    model="LSTM",
    hidden_dim=64,
    batch_size=1,
    n_epochs=100,
    likelihood=LaplaceLikelihood(),
    optimizer_kwargs={"lr": 1e-3},
    model_name="tipping_lstm_tune",
    log_tensorboard=False,
    random_state=42,
    input_chunk_length=25,
    output_chunk_length=75,
    torch_device_str = "cuda:0",
)

parameters = {
'hidden_dim':[20,30],
'batch_size':[16,32],
'dropout':[0,0.2],
'n_rnn_layers':[1,2],
'input_chunk_length':[25],
'output_chunk_length':[75],
}

my_model.gridsearch(parameters=parameters, series=series, use_fitted_values=True, start=25)
