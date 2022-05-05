from darts.models import StatsForecastAutoARIMA, KalmanForecaster, NaiveDrift
from darts import TimeSeries
import sys
sys.path.append("../")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
import torch
from utils import get_train_series, truth_dist, make_df
import argparse
from functools import reduce
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--forecasting_model",
    default="arima",
    type=str,
    help="model to train with (arima)",
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

train_series = get_train_series(args)

_model = {"arima" : StatsForecastAutoARIMA,}[args.forecasting_model]

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
    else:
        t_series = train_series[-input_len:]
    
    # truth_dist 
    start_t = input_len
    t_dist = truth_dist(args.sim_model, t_series, input_len, output_len, n_draws=n_draws, reverse=args.decrease, start_t=start_t)
    
    t_max = 250-input_len if args.sim_model != "hopf" else 100
    
    ensemble_preds = []
    if args.sim_model != "hopf":
        model = _model().fit(series=t_series)
        _preds = model.predict(t_max, num_samples=n_draws)
        ensemble_preds.append(_preds)
        ensemble_series = reduce(lambda a, b: a.concatenate(b, axis="sample"), ensemble_preds)
    else:
        h_t_series, p_t_series = t_series.univariate_component(0), t_series.univariate_component(1)
        for series in [h_t_series, p_t_series]:
            model = _model().fit(series=series)
            _preds = model.predict(t_max, num_samples=n_draws)
            ensemble_preds.append(_preds)
      
        ensemble_series = reduce(lambda a, b: a.concatenate(b, axis="component"), ensemble_preds)
    
    case = "tipped" if args.tipped else "nontipped"
        
    df = make_df(ensemble_series, t_dist, t_series, args.sim_model, args.forecasting_model.lower(), case, args.n_samples, i)
    final_df = final_df.append(df, ignore_index=True)
    
final_df.to_csv(f"forecasts/reps={args.n_samples}/{args.output_file_name}.csv.gz", index=False)
