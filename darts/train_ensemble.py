import pandas as pd
import numpy as np
import os
import sys
sys.path.append("../")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from darts import TimeSeries
from pandas import DataFrame
import torch
from darts.models import RNNModel, BlockRNNModel
from darts.utils.likelihood_models import LaplaceLikelihood
from utils import preprocessed_t_series, truth_dist, import_hypers, make_df
import argparse
import yaml
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="lstm",
    type=str,
    help="model to train with (lstm or block_rnn)",
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
    "-e",
    "--evaluate",
    action="store_true",
    help="Flag whether to plot and save a csv of samples",
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
    "--hyper_file",
    default="lstm",
    help="Select which hyperparameter file to load hypers from",
)
parser.add_argument(
    "-c",
    "--case",
    default="none",
    type=str,
    help="Special cases to consider (none, tipped, non_tipped, both)",
)
args = parser.parse_args()


if args.hyper_file == "stochastic_lstm":
    from train_hyperparams.stochastic_lstm import hyperparameters
elif args.hyper_file == "saddle_lstm":
    from train_hyperparams.saddle_lstm import hyperparameters
elif args.hyper_file == "stochastic_block":
    from train_hyperparams.stochastic_block import hyperparameters
elif args.hyper_file == "saddle_block":
    from train_hyperparams.saddle_block import hyperparameters

model = {"lstm" : RNNModel, "block_rnn" : BlockRNNModel}[args.model.lower()]

models = []
for i in range(42, 47):
    hyperparameters["random_state"] = i
    np.random.seed(i)
    
    # Selecting case to pick training set
    if args.case.lower() == "tipped" and args.tp_model == "stochastic":
        train_series = TimeSeries.from_csv("stochastic_tipped.csv", time_col="time")
    elif args.case.lower() == "nontipped" and args.tp_model == "stochastic":
        train_series = TimeSeries.from_csv("stochastic_non_tipped.csv", time_col="time")
    elif args.case.lower() == "both" and args.tp_model == "stochastic":
        train_series = TimeSeries.from_csv("stochastic_tipped.csv", time_col="time")
        train_series_ = TimeSeries.from_csv("stochastic_non_tipped.csv", time_col="time")
        train_series = train_series.concatenate(train_series_, axis="sample")
    elif args.case.lower() == "none":
        # Generating time series
        train_series, _ = preprocessed_t_series(args.tp_model, args.n_samples)
        

    my_model = model(
        likelihood=LaplaceLikelihood(),
        **hyperparameters
    )
    
    my_model.fit(
        train_series,
        verbose=True,
    )
    
    models.append(my_model)
    del my_model

# Generating time series
if args.evaluate:
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    if not os.path.exists(f"plots/ensemble_{args.output_file_name}/"):
        os.makedirs(f"plots/ensemble_{args.output_file_name}/")
    if not os.path.exists("forecasts/"):
            os.makedirs("forecasts/")
    input_len = hyperparameters["input_chunk_length"]
    output_len = hyperparameters["output_chunk_length"]
    final_df = DataFrame()
  
    for i in range(5):
        train_series, _ = preprocessed_t_series(args.tp_model, 1)
        train_series[:input_len].plot(color="blue", label='truth')
        t_series, v_series = train_series.split_before(input_len)
        t_dist = truth_dist(args.tp_model, t_series, input_len, output_len, n_samples=100)
        
        t_max = 250-input_len
        
        t_dist.plot(low_quantile=0.025, high_quantile=0.975, color="blue", label="_nolegend_", linestyle="dotted")
        ensemble_preds = []
        for model in models:
            ensemble_preds.append(model.predict(t_max, t_series, num_samples=100))
        ensemble_series = reduce(lambda a, b: a.concatenate(b, axis="sample"), ensemble_preds)
        
        ensemble_series.plot(low_quantile=0.025, high_quantile=0.975, linestyle="dotted", color="orange", label='prediction')
        df = make_df(ensemble_series, t_dist, args.tp_model, args.model.lower(), args.case, args.n_samples)
        df["iter"] = i
        final_df = final_df.append(df, ignore_index=True)
        plt.savefig(f"plots/ensemble_{args.output_file_name}/{i}")
        plt.clf()
        
    final_df.to_csv(f"forecasts/{args.output_file_name}.csv.gz")
