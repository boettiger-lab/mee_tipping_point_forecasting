import stan
import cmdstanpy
import sys
sys.path.append("../darts/")
from utils import preprocessed_t_series
import numpy as np
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tp_model",
    default="stochastic",
    type=str,
    help="tp model to train with (stochastic/saddle)",
)
args = parser.parse_args()

sm = cmdstanpy.CmdStanModel(stan_file=f"{args.tp_model.lower()}_tp.stan")

_, _data = preprocessed_t_series(f"{args.tp_model.lower()}", 100)
data = dict(n=100, t_max=250, x=_data)
# Sample using Stan
samples = sm.sample(
    data=data,
    chains=4,
    iter_sampling=10000, 
    iter_warmup=10000,
    seed=42,
)

print(samples.summary())
import pdb; pdb.set_trace()
