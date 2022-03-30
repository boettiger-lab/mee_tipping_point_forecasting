import pandas as pd
import numpy as np
import os
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
    help="model to train with (lstm, tcn or transformer)",
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
    "-p",
    "--plot",
    action="store_true",
    help="Flag whether to plot",
)
args = parser.parse_args()

np.random.seed(42)

if args.model == "lstm":
    from train_hyperparams.lstm import hyperparameters
elif args.model == "tcn":
    from train_hyperparams.tcn import hyperparameters
elif args.model == "transformer":
    from train_hyperparams.transformer import hyperparameters

model = {"lstm" : RNNModel, "tcn" : TCNModel, "transformer" : TransformerModel}[args.model.lower()]
# Generating time series
train_series = preprocessed_t_series(args.n_samples)

#TO DO: make a list of kwargs to then unpack in model call

my_model = model(
    likelihood=LaplaceLikelihood(),
    **hyperparameters
)


my_model.fit(
    train_series,
    verbose=True,
)

# Generating time series
if args.plot:
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    for i in range(3):
        train_series = preprocessed_t_series(1)
        train_series[:100].plot(label="truth")
        t_series, v_series = train_series.split_before(25)
        preds = my_model.predict(75, t_series, num_samples=10000)
        preds.plot(low_quantile=0.15, high_quantile=0.85, label="predict 15-85th percentiles")
        plt.savefig(f"plots/{args.output_file_name}_{i}")
        plt.clf()
