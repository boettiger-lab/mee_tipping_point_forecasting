import pandas as pd
import numpy as np
import os
import sys
sys.path.append("../")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from darts import TimeSeries
import torch
from darts.models import RNNModel, TCNModel, TransformerModel
from darts.utils.likelihood_models import LaplaceLikelihood
from utils import preprocessed_t_series, truth_dist
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
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Seed selection",
)
parser.add_argument(
    "-t",
    "--tp_model",
    default="stochastic",
    type=str,
    help="Select which model to use for the tipping point (stochastic/saddle)",
)
parser.add_argument(
    "--random_alpha",
    action="store_true",
    help="Select whether to use a random alpha or not on saddle node tp",
)
args = parser.parse_args()
np.random.seed(args.seed)

if args.model == "lstm":
    from train_hyperparams.lstm import hyperparameters
elif args.model == "tcn":
    from train_hyperparams.tcn import hyperparameters
elif args.model == "transformer":
    from train_hyperparams.transformer import hyperparameters

hyperparameters["random_state"] = args.seed

model = {"lstm" : RNNModel, "tcn" : TCNModel, "transformer" : TransformerModel}[args.model.lower()]
# Generating time series
train_series = preprocessed_t_series(args, args.n_samples)

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
    colors = ["blue", "green", "orange"]
    input_len = hyperparameters["input_chunk_length"]
    output_len = hyperparameters["output_chunk_length"]
    for i in range(3):
        torch.manual_seed(i+100)
        np.random.seed(i+100)
        train_series = preprocessed_t_series(args, 1)
        train_series[:input_len].plot(color="blue", label='truth')
        t_series, v_series = train_series.split_before(input_len)
        t_dist = truth_dist(args.tp_model, t_series, input_len, output_len, n_samples=100, random_alpha=args.random_alpha)
        
        t_max = 250-input_len
        
        t_dist.plot(low_quantile=0.025, high_quantile=0.975, color="blue", label="_nolegend_", linestyle="dotted")
        preds = my_model.predict(t_max, t_series, num_samples=1000)
        preds.plot(low_quantile=0.025, high_quantile=0.975, linestyle="dotted", color="orange", label='prediction')

        plt.savefig(f"plots/{args.output_file_name}_{i}")
        plt.clf()
