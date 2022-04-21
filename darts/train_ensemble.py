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
from darts.models import RNNModel, BlockRNNModel, TransformerModel
from darts.utils.likelihood_models import LaplaceLikelihood
from utils import preprocessed_t_series, truth_dist, make_df, get_train_series, plot
import argparse
import yaml
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="lstm",
    type=str,
    help="model to train with (lstm/block_rnn/transformer/gru)",
)
parser.add_argument(
    "-s",
    "--n_samples",
    default=100,
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
    help="Select which model to use for the tipping point (stochastic/saddle/hopf)",
)
parser.add_argument(
    "-c",
    "--case",
    default="none",
    type=str,
    help="Special cases to consider (none, tipped, non_tipped, both)",
)
parser.add_argument(
    "--reverse",
    action="store_true",
    help="Flag to use decreasing Hopf model",
)
args = parser.parse_args()

if args.model == "lstm": 
    if args.tp_model == "stochastic":
        from train_hyperparams.stochastic_lstm import hyperparameters
    elif args.tp_model == "saddle":
        from train_hyperparams.saddle_lstm import hyperparameters
    elif args.tp_model == "hopf":
        from train_hyperparams.hopf_lstm import hyperparameters
elif args.model == "gru":
    if args.tp_model == "stochastic":
        from train_hyperparams.stochastic_gru import hyperparameters
    elif args.tp_model == "saddle":
        from train_hyperparams.saddle_gru import hyperparameters
    elif args.tp_model == "hopf":
        from train_hyperparams.hopf_gru import hyperparameters
elif args.model == "block_rnn":
    if args.tp_model == "stochastic":
        from train_hyperparams.stochastic_block import hyperparameters
    elif args.tp_model == "saddle":
        from train_hyperparams.saddle_block import hyperparameters
    elif args.tp_model == "hopf":
        from train_hyperparams.hopf_block import hyperparameters
elif args.model == "transformer":
    if args.tp_model == "stochastic":
        from train_hyperparams.stochastic_transformer import hyperparameters
    elif args.tp_model == "saddle":
        from train_hyperparams.saddle_transformer import hyperparameters
    elif args.tp_model == "hopf":
        from train_hyperparams.hopf_transformer import hyperparameters

model = {"lstm" : RNNModel, "gru" : RNNModel, "block_rnn" : BlockRNNModel, "transformer" : TransformerModel}[args.model.lower()]

models = []
for i in range(42, 47):
    hyperparameters["random_state"] = i
    np.random.seed(i)
    
    train_series = get_train_series(args)

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
        np.random.seed(i)
        torch.manual_seed(i)
        
        train_series = preprocessed_t_series(args.tp_model, 1)
        
        if args.tp_model == "hopf":
            x, y = train_series[:input_len].univariate_component(0), train_series[:input_len].univariate_component(1)
            x.plot(color="blue", label='truth')
            y.plot(color="gold", label="truth")
        else:
            train_series[:input_len].plot(color="blue", label='truth')
            
        t_series = train_series[:input_len]
        t_dist = truth_dist(args.tp_model, t_series, input_len, output_len, n_samples=100, reverse=args.reverse)
        
        t_max = 250-input_len if args.tp_model != "hopf" else 200-input_len
        plot(args.tp_model, t_dist, "blue", "gold")
        
        ensemble_preds = []
        for model in models:
            ensemble_preds.append(model.predict(t_max, t_series, num_samples=100))
        ensemble_series = reduce(lambda a, b: a.concatenate(b, axis="sample"), ensemble_preds)
        
        plot(args.tp_model, ensemble_series, "orange", "green")
        df = make_df(ensemble_series, t_dist, args.tp_model, args.model.lower(), args.case, args.n_samples)
        df["iter"] = i
        final_df = final_df.append(df, ignore_index=True)
        plt.savefig(f"plots/ensemble_{args.output_file_name}/{i}")
        plt.clf()
        
    final_df.to_csv(f"forecasts/{args.output_file_name}.csv.gz", index=False)


