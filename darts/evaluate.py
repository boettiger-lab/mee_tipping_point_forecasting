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


# NB Cannot use Hopf model here as we don't have the data scaler,
# so have to retrain these ones to evaluate.
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
    default=1,
    type=int,
    help="Number of samples that the model was trained on",
)
parser.add_argument(
    "-o",
    "--output_file_name",
    default="trash",
    type=str,
    help="File name of plots",
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
    help="Use model trained on tipped or nontipped data",
)
parser.add_argument(
    "--decrease",
    action="store_true",
    help="Use model trained on decreasing or increasing hopf model",
)
args = parser.parse_args()


_model = {"lstm" : RNNModel, "gru" : RNNModel, "block_rnn" : BlockRNNModel, "transformer" : TransformerModel}[args.forecasting_model.lower()]
models = []
for i in range(42, 47):
    model_name = f"{args.forecasting_model}_{args.sim_model}_{args.n_samples}_{args.decrease}_{args.tipped}_{i}"
    model = _model.load_from_checkpoint(f"{model_name}", file_name="last-epoch=499.ckpt")
    models.append(model)

if not os.path.exists("forecasts/"):
    os.makedirs("forecasts/")
    
input_len = 25
output_len = 24
final_df = DataFrame()

if not os.path.exists("forecasts/"):
    os.makedirs("forecasts/")
input_len = 25
output_len = 24
final_df = DataFrame()
for i in range(5):
    np.random.seed(i)
    torch.manual_seed(i)
    n_draws = 100
    
    args.n_samples = 1
    train_series = get_train_series(args)
    
    if args.sim_model == "stochastic":
        t_series = train_series[:input_len]
    elif args.sim_model == "saddle":
        t_series = train_series[-input_len:]
    
    # truth_dist 
    start_t = input_len
    t_dist = truth_dist(args.sim_model, t_series, input_len, output_len, n_draws=n_draws, reverse=args.decrease, start_t=start_t)
    
    t_max = 250-input_len
    
    ensemble_preds = []
    for model in models:
        _preds = model.predict(t_max, t_series, num_samples=n_draws)
        ensemble_preds.append(_preds)
    ensemble_series = reduce(lambda a, b: a.concatenate(b, axis="sample"), ensemble_preds)
    
    case = "tipped" if args.tipped else "nontipped"
        
    df = make_df(ensemble_series, t_dist, t_series, args.sim_model, args.forecasting_model.lower(), case, args.n_samples, i)
    final_df = final_df.append(df, ignore_index=True)
    
final_df.to_csv(f"forecasts/reps={args.n_samples}/{args.output_file_name}.csv.gz", index=False)
