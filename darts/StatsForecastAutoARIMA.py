from darts.models import StatsForecastAutoARIMA
from darts import TimeSeries
import sys
sys.path.append("../")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import preprocessed_t_series, truth_dist
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--tp_model",
    default="stochastic",
    type=str,
    help="Select which model to use for the tipping point (stochastic/saddle)",
)
parser.add_argument(
    "--case",
    default="tipped",
    type=str,
    help="Select which case for the training series",
)
args = parser.parse_args()

if args.case.lower() == "tipped" and args.tp_model == "stochastic":
    train_series = TimeSeries.from_csv("stochastic_tipped.csv", time_col="time")
elif args.case.lower() == "nontipped" and args.tp_model == "stochastic":
    train_series = TimeSeries.from_csv("stochastic_non_tipped.csv", time_col="time")
elif args.case.lower() == "both" and args.tp_model == "stochastic":
    train_series = TimeSeries.from_csv("stochastic_tipped.csv", time_col="time")
    train_series_ = TimeSeries.from_csv("stochastic_non_tipped.csv", time_col="time")
    train_series = train_series.concatenate(train_series_, axis="sample")
        
input_len =25
output_len=24
for i in range(3):
    train_series[:input_len].plot(color="blue", label='truth')
    t_series, v_series = train_series.split_before(input_len)
    t_dist = truth_dist("stochastic", t_series, input_len, output_len, n_samples=100)
    
    t_max = 250-input_len
    
    t_dist.plot(low_quantile=0.025, high_quantile=0.975, color="blue", label="_nolegend_", linestyle="dotted")
    model = StatsForecastAutoARIMA().fit(t_series)
    preds = model.predict(t_max, num_samples=100)
    preds.plot(low_quantile=0.025, high_quantile=0.975, linestyle="dotted", color="orange", label='prediction')
    
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    plt.savefig(f"plots/StatsForecastAutoARIMA_{i}")
    plt.clf()
