from darts.models import ExponentialSmoothing
from darts.utils.utils import ModelMode
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
    "--random_alpha",
    action="store_true",
    help="Select whether to use a random alpha or not on saddle node tp",
)
args = parser.parse_args()

train_series = preprocessed_t_series(args, 100)

model = ExponentialSmoothing(trend=ModelMode.NONE, seasonal=None, seasonal_periods=0)

input_len =25
output_len=24
for i in range(3):
    train_series = preprocessed_t_series(args, 1)
    train_series[:input_len].plot(color="blue", label='truth')
    t_series, v_series = train_series.split_before(input_len)
    t_dist = truth_dist("stochastic", t_series, input_len, output_len, n_samples=100, random_alpha=args.random_alpha)
    
    t_max = 250-input_len
    
    t_dist.plot(low_quantile=0.025, high_quantile=0.975, color="blue", label="_nolegend_", linestyle="dotted")
    preds = model.historical_forecasts(train_series, num_samples=1000, start=3, forecast_horizon=24, retrain=True)
    preds.plot(low_quantile=0.025, high_quantile=0.975, linestyle="dotted", color="orange", label='prediction')
    train_series.plot(color="blue", label="_nolegend_")

    plt.savefig(f"plots/es_test_{i}")
    plt.clf()
