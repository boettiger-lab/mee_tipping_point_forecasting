import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import tipping_point
from darts import TimeSeries
from darts.models import RNNModel, TCNModel, TransformerModel
from darts.utils.likelihood_models import LaplaceLikelihood
from utils import preprocessed_t_series
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="lstm",
    type=str,
    help="model to train with (lstm, tcm or transformer)",
)
parser.add_argument(
    "-s",
    "--n_samples",
    default=1000,
    type=int,
    help="# of samples to train on",
)
parser.add_argument(
    "-o",
    "--output_file_name",
    default="trash",
    type=str,
    help="File name of plots",
)
parser.add_argument(
    "-f",
    "--parameter_yaml",
    default="train",
    type=str,
    help="File name of YAML used to store parameters",
)
args = parser.parse_args()


with open(f'{args.parameter_yaml}.yaml') as fh:
    params_yaml = yaml.load(fh, Loader=yaml.FullLoader)

model = [*params_yaml][0]
PARAMETERS = params_yaml[model]
model = model.lower()
# 
if model == "lstm":
    PARAMETERS["model"] = "LSTM"
if "optimizer_kwargs" in PARAMETERS.keys():
    PARAMETERS["optimizer_kwargs"] = PARAMETERS["optimizer_kwargs"][0]

model = {"lstm" : RNNModel, "tcn" : TCNModel, "transformer" : TransformerModel}[args.model.lower()]
# Generating time series
train_series = preprocessed_t_series(args.n_samples)

#TO DO: make a list of kwargs to then unpack in model call

my_model = model(
    likelihood=LaplaceLikelihood(),
    **PARAMETERS
)


my_model.fit(
    train_series,
    verbose=True,
)

# Generating time series
# Generating time series
for i in range(3):
    train_series = preprocessed_t_series(1)
    t_series, v_series = train_series.split_before(101)
    t_series.plot(label="truth")
    preds = my_model.historical_forecasts(t_series, retrain=False, start=25, num_samples=10000, forecast_horizon=75)
    preds.plot(low_quantile=0.15, high_quantile=0.85, label="historical 15-85th percentiles")
    
    t_series, v_series = t_series.split_before(25)
    preds = my_model.predict(75, t_series, num_samples=10000)
    preds.plot(low_quantile=0.15, high_quantile=0.85, label="predict 15-85th percentiles")
    plt.savefig(f"plots/{args.output_file_name}_{i}")
    plt.clf()
