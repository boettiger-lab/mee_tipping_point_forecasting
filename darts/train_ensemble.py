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
from utils import preprocessed_t_series, truth_dist, make_df, get_train_series
import argparse
from functools import reduce
from darts.dataprocessing.transformers import Scaler

scaler = Scaler()
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--forecasting_model",
    default="lstm",
    type=str,
    help="model to train with (lstm/block_rnn/transformer/gru)",
)
parser.add_argument(
    "-s",
    "--n_samples",
    default=10,
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
    "--sim_model",
    default="stochastic",
    type=str,
    help="Select which dynamics to use for the tipping point (stochastic/saddle/hopf)",
)
parser.add_argument(
    "--tipped",
    action="store_true",
    help="Whether to use the tipped or nontipped training set",
)
parser.add_argument(
    "--decrease",
    action="store_true",
    help="Flag to use decreasing Hopf model",
)
args = parser.parse_args()

if args.forecasting_model == "lstm": 
    if args.sim_model == "stochastic":
        from train_hyperparams.stochastic_lstm import hyperparameters
    elif args.sim_model == "saddle":
        from train_hyperparams.saddle_lstm import hyperparameters
    elif args.sim_model == "hopf":
        if args.decrease:
            from train_hyperparams.hopf_decrease_lstm import hyperparameters
        else:
            from train_hyperparams.hopf_lstm import hyperparameters
elif args.forecasting_model == "gru":
    if args.sim_model == "stochastic":
        from train_hyperparams.stochastic_gru import hyperparameters
    elif args.sim_model == "saddle":
        from train_hyperparams.saddle_gru import hyperparameters
    elif args.sim_model == "hopf":
        if args.decrease:
            from train_hyperparams.hopf_decrease_gru import hyperparameters
        else:
            from train_hyperparams.hopf_gru import hyperparameters
elif args.forecasting_model == "block_rnn":
    if args.sim_model == "stochastic":
        from train_hyperparams.stochastic_block import hyperparameters
    elif args.sim_model == "saddle":
        from train_hyperparams.saddle_block import hyperparameters
    elif args.sim_model == "hopf":
        if args.decrease:
            from train_hyperparams.hopf_decrease_block import hyperparameters
        else:
            from train_hyperparams.hopf_block import hyperparameters
elif args.forecasting_model == "transformer":
    if args.sim_model == "stochastic":
        from train_hyperparams.stochastic_transformer import hyperparameters
    elif args.sim_model == "saddle":
        from train_hyperparams.saddle_transformer import hyperparameters
    elif args.sim_model == "hopf":
        if args.decrease:
            from train_hyperparams.hopf_decrease_transformer import hyperparameters
        else:
            from train_hyperparams.hopf_transformer import hyperparameters

model = {"lstm" : RNNModel, "gru" : RNNModel, "block_rnn" : BlockRNNModel, "transformer" : TransformerModel}[args.forecasting_model.lower()]

models = []
for i in range(42, 47):
    hyperparameters["random_state"] = i
    np.random.seed(i)
    
    train_series = get_train_series(args)
    if args.sim_model == "hopf":
        train_series = scaler.fit_transform(train_series)
    hyperparameters["model_name"] = f"{args.forecasting_model}_{args.sim_model}_{args.n_samples}_{args.decrease}_{args.tipped}_{i}"
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
    if not os.path.exists("forecasts/"):
            os.makedirs("forecasts/")
    input_len = hyperparameters["input_chunk_length"]
    output_len = hyperparameters["output_chunk_length"]
    final_df = DataFrame()
  
    for i in range(5):
        np.random.seed(i)
        torch.manual_seed(i)
        n_draws = 100
        
        train_series = preprocessed_t_series(args.sim_model, 1)
        
        if args.sim_model == "hopf":
            t_series = train_series[-input_len:]
        else:
            t_series = train_series[:input_len]
        
        # truth_dist 
        start_t = input_len if args.sim_model != "hopf" else 100
        t_dist = truth_dist(args.sim_model, t_series, input_len, output_len, n_draws=n_draws, reverse=args.decrease, start_t=start_t)
        
        t_max = 250-input_len if args.sim_model != "hopf" else 100
        
        ensemble_preds = []
        for model in models:
            if args.sim_model == "hopf":
                _preds = scaler.inverse_transform(model.predict(t_max, t_series, num_samples=n_draws))
            else: 
                _preds = model.predict(t_max, t_series, num_samples=n_draws)
            ensemble_preds.append(_preds)
        ensemble_series = reduce(lambda a, b: a.concatenate(b, axis="sample"), ensemble_preds)
        
        if args.sim_model == "stochastic" or args.sim_model == "saddle":
            case = "tipped" if args.tipped else "nontipped"
        elif args.sim_model == "hopf":
            case = "decrease" if args.decrease else "increase"
            
        df = make_df(ensemble_series, t_dist, t_series, args.sim_model, args.forecasting_model.lower(), case, args.n_samples, i)
        final_df = final_df.append(df, ignore_index=True)
        
    final_df.to_csv(f"forecasts/{args.output_file_name}.csv.gz", index=False)
