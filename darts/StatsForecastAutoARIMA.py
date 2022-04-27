from darts.models import StatsForecastAutoARIMA, KalmanForecaster, NaiveDrift
from darts import TimeSeries
import sys
sys.path.append("../")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import preprocessed_t_series, truth_dist, make_df
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--sim_model",
    default="stochastic",
    type=str,
    help="Select which model to use for the tipping point (stochastic/saddle/hopf)",
)
parser.add_argument(
    "--case",
    default="tipped",
    type=str,
    help="Select which case for the training series",
)
parser.add_argument(
    "-f",
    "--forecasting_model",
    default="arima",
    type=str,
    help="Select which model to train",
)
parser.add_argument(
    "-o",
    "--output_file_name",
    default="trash",
    type=str,
    help="Select which filename to use",
)
args = parser.parse_args()

if args.case.lower() == "tipped" and args.sim_model == "stochastic":
    train_series = TimeSeries.from_csv("stochastic_tipped.csv", time_col="time")
elif args.case.lower() == "nontipped" and args.sim_model == "stochastic":
    train_series = TimeSeries.from_csv("stochastic_non_tipped.csv", time_col="time")
elif args.case.lower() == "both" and args.sim_model == "stochastic":
    train_series = TimeSeries.from_csv("stochastic_tipped.csv", time_col="time")
    train_series_ = TimeSeries.from_csv("stochastic_non_tipped.csv", time_col="time")
    train_series = train_series.concatenate(train_series_, axis="sample")

_model = {"arima" : StatsForecastAutoARIMA,}[args.forecasting_model]
input_len =25
output_len=24
if not os.path.exists("forecasts/"):
    os.makedirs("forecasts/")

for i in range(1):
    t_series, v_series = train_series.split_before(input_len)
    t_dist = truth_dist("stochastic", t_series, input_len, output_len, n_draws=100)
    
    t_max = 250-input_len
    
    model = _model().fit(t_series)
    preds = model.predict(t_max, num_samples=100)
    
    df = make_df(preds, t_dist, t_series, args.sim_model, args.forecasting_model.lower(), args.case, 1, 0)
    df.to_csv(f"forecasts/{args.output_file_name}.csv.gz", index=False)
