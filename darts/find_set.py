
import numpy as np
from utils import preprocessed_t_series, count_tipped
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--sim_model",
    default="stochastic",
    type=str,
    help="Select which model to use for the tipping point (stochastic/saddle/hopf)",
)
args = parser.parse_args()

if args.sim_model == "stochastic":
    flag_non = False
    flag_tip = False
    while True:
        series = preprocessed_t_series(args.sim_model, 1)
        vals = series.all_values()
        
        if count_tipped(vals) == 0:
            series.to_csv(f"{args.sim_model}_nontipped.csv")
            flag_non = True
    
        if count_tipped(vals) == 1:
            series.to_csv(f"{args.sim_model}_tipped.csv")
            flag_tip = True
        
        if flag_non and flag_tip:
            break
