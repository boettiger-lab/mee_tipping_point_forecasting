import stan
import cmdstanpy
import sys
sys.path.append("../darts/")
from utils import preprocessed_t_series
import numpy as np

sm = cmdstanpy.CmdStanModel(stan_file="stochastic_tp.stan")

np.random.seed(42)
_, _data = preprocessed_t_series("stochastic", 100)
data = dict(n=100, t_max=250, x=_data.reshape(100, 250))
# Sample using Stan
samples = sm.sample(
    data=data,
    chains=4,
    iter_sampling=10000, 
    iter_warmup=5000,
)

print(samples.summary())
import pdb; pdb.set_trace()
