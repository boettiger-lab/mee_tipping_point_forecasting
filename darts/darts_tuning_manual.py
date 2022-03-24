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
import random

N_TRIALS = 10

parameters = {
'hidden_dim':[64, 132, 256],
'batch_size':[1, 8, 16, 32,64,128, 256],
'dropout':[0,0.2,0.5],
'n_rnn_layers':[1,2],
'random_state': [42],
'optimizer_kwargs' : [{"lr": 1e-3}, {"lr": 1e-4}, {"lr": 1e-5}, {"lr": 1e-2}]
}

# Generating time series
train_series = preprocessed_t_series(100)

for i in range(N_TRIALS):
    # Randomly selecting a parameter dictionary
    parameter_keys = list(parameters.keys())
    trial_parameters = {
        parameter_keys[j] : random.choice(parameters[parameter_keys[j]]) for j in range(len(parameters))
    }
    
    # Making a file name suffix to keep track of hyperparams
    trial_parameter_keys = list(trial_parameters.keys())
    trial_string = ""
    for k in range(len(parameters)):
        trial_string += f"_{trial_parameter_keys[k]}_{trial_parameters[trial_parameter_keys[k]]}"

    my_model = RNNModel(
        model="LSTM",
        n_epochs=100,
        likelihood=LaplaceLikelihood(),
        model_name=f"lstm_tuning{trial_string}",
        log_tensorboard=True,
        input_chunk_length=25,
        output_chunk_length=75,
        torch_device_str = "cuda:0",
        **trial_parameters
    )
    
    my_model.fit(
    train_series,
    verbose=True,
    )
